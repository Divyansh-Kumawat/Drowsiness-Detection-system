#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Load cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Counters for eye detection
eyes_closed_counter = 0
eyes_open_counter = 0

def detect_drowsiness(eyes_detected, min_eyes=2):
    """
    Simple drowsiness detection based on eye detection
    """
    global sleep, drowsy, active, status, color, eyes_closed_counter, eyes_open_counter
    
    if len(eyes_detected) < min_eyes:
        # Eyes not detected or closed
        eyes_closed_counter += 1
        eyes_open_counter = 0
        
        if eyes_closed_counter > 10:  # Eyes closed for more than 10 frames
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 3:
                status = "SLEEPING !!!"
                color = (0, 0, 255)  # Red
        elif eyes_closed_counter > 5:  # Eyes closed for 5-10 frames
            drowsy += 1
            sleep = 0
            active = 0
            if drowsy > 3:
                status = "Drowsy !"
                color = (0, 165, 255)  # Orange
    else:
        # Eyes detected (open)
        eyes_open_counter += 1
        eyes_closed_counter = 0
        
        if eyes_open_counter > 5:
            active += 1
            drowsy = 0
            sleep = 0
            if active > 3:
                status = "Active :)"
                color = (0, 255, 0)  # Green

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize face_frame for cases when no face is detected
    face_frame = frame.copy()

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(face_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for eyes (upper half of face)
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = face_frame[y:y+h//2, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Analyze drowsiness based on eye detection
        detect_drowsiness(eyes)
        
        # Display status on frame
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Display eye count for debugging
        cv2.putText(frame, f"Eyes detected: {len(eyes)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frames
    cv2.imshow("Driver Drowsiness Detection", frame)
    cv2.imshow("Face Detection", face_frame)
    
    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
