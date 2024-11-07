import cv2
import mediapipe as mp
import numpy as np
import time

#Tracks points centered at the shoulder elbow and wrist and computes the angle at the elbow or the angle being created at the hips 
def calculate_angle(a,b,c):
    a = np.array(a) #Shoulder
    b = np.array(b) #Elbow
    c = np.array(c) #Wrist

    #calculates angle and converts it from radians to degrees
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) 
    angle = np.abs(radians*(180/np.pi))

    #Exures that angle is in between 0 and 180 which represents limit for arm movement
    if angle > 180:
        angle = 360 - angle
    
    return angle

#Write a function that can choose/sense between different workouts that the user is trying to achieve.
#Work with the visibility of each part of the body to try and tell


#Gives drawing utilities used to visualize poses
mp_drawing = mp.solutions.drawing_utils
#Provides pose estimation models
mp_pose = mp.solutions.pose

#Video Capture portion of the code

#Sets up capture device, eg: Webcam, 0 parameter represents webcam
cap = cv2.VideoCapture(0)
#Sets up a new mediapipe instance, the two parameters are specifying the detection and tracking confidence values
with mp_pose.Pose(min_detection_confidence = 0.6, min_tracking_confidence = 0.6) as pose:
    while(cap.isOpened()):
        #frame holds the capture
        ret, frame = cap.read()

        #Recolors image, because color format should be formatted in RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Makes the detection, and stores it in results(an array)
        results = pose.process(image)

        #Make image writable again
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display

            #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #Grabs the x and y values of the 3 points that are needed for angle calculation
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]


            #position of left and right eye
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            #Calculates middle of head for displaying messages conviniently.
            mid_x = (right_eye[0] + left_eye[0]) / 2
            mid_y = (right_eye[1] + left_eye[1]) / 2
            position_to_display = (mid_x,mid_y)
            message = "Please keep elbow in frame"

            #hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            #knee = [andmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

            #Calcultates the angle at elbow region
            angle = calculate_angle(shoulder,elbow,wrist)
            #Calculates the angle between shoulder hip and knee
            #angle2 = calculate_angle(shoulder,hip,knee)

            #Multiplies the elbow coordinates with the camera dimensions - image.shape[1] and [0] find the correct
            #dimensions of device's webcam
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        
            cv2.putText(image, message, position_to_display, cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        #Drawings the detection on the capture, the landmarks represent coordingates, and the connections are the different joints
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))          

        #Pop up that allows visualization named "Mediapipe Feed"
        cv2.imshow('Mediapipe Feed', image)

        #Either hitting q on keyboard, exit the loop
        if(cv2.waitKey(10) & 0XFF == ord('q')):
            break

#Releases webcam
cap.release()
#Closes the video feed
cv2.destroyAllWindows()

