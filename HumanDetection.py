import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# Initialize the HOG descriptor for pedestrian detection
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

while True:
    #Reading the Frame
    ret, frame = cap.read()

    #Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect pedestrians in the grayscale frame using HOG
    rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Iterate over the detected rectangles and their corresponding confidence levels
    for (x, y, w, h), confidence in zip(rects, weights):

        # Draw a rectangle around the detected pedestrian
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        # Add a label indicating the confidence level of the detection
        label = f"Person: {confidence}"
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #print No of Humans in Console
    num_humans = len(rects)
    print("Number of Humans:", num_humans)

    # Display the frame with detections
    cv.imshow('Camera Feed', frame)
    #break condition to exit the loop 
    if cv.waitKey(1) == ord('q'):
        break
    
# Release the camera and destroy all windows
cap.release()
cv.destroyAllWindows()
