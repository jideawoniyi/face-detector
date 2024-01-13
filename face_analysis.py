#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 02:10:56 2023

@author: Jide Awoniyi
"""

import cv2
import dlib
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image

# Step 1: Install required libraries
# pip install opencv-python-headless dlib tensorflow keras

# Step 2: Load the necessary models
face_detect = dlib.get_frontal_face_detector()


predictor_path = "/Users/Smartsys/Documents/Face Detector/shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(predictor_path)


emotion_classifier = load_model("/Users/Smartsys/Documents/Face Detector/emotion_recognition_model.h5")
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def analyze_face(frame, face, landmarks):
    # Step 5: Extract features and objects from the landmarks
    description = []

    # Check for smile
    mouth_ratio = (landmarks.part(66).y - landmarks.part(62).y) / (landmarks.part(64).x - landmarks.part(60).x)
    if mouth_ratio > 0.3:
        description.append("smiling")

    # Check for cap
    forehead_height = (landmarks.part(27).y - landmarks.part(24).y)
    if forehead_height < 20:
        description.append("wearing a cap")

    # Check for earrings
    ear_height = landmarks.part(45).y - landmarks.part(36).y
    if ear_height > 80:
        description.append("wearing earrings")

    # Emotion recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[face.top():face.bottom(), face.left():face.right()]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    description.append(f"Emotion: {label}")

    return description

def main():
    # Step 3: Access the camera and process the video frames
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 4: Detect faces and facial landmarks
        faces = face_detect(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Analyze face
            description = analyze_face(frame, face, landmarks)
            print(description)

            # Draw rectangle around the face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        # Show the video frame
        cv2.imshow("Face Analysis", frame)

        # Press 'q' to quit the app
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
