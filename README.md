
# Face Analysis and Emotion Recognition Project

## Overview
This project includes two Python scripts:
1. `face_analysis.py` - A script for real-time face analysis and emotion recognition using OpenCV, Dlib, and Keras.
2. `train_emotion_model.py` - A script for training an emotion recognition model using a Convolutional Neural Network (CNN) on the FER2013 dataset.

## Requirements
- Python 3
- Libraries: OpenCV, Dlib, Keras, TensorFlow, NumPy, Pandas, Scikit-Learn

## Installation
To install the required libraries, run the following command:
```
pip install opencv-python-headless dlib tensorflow keras numpy pandas scikit-learn
```

## File Descriptions

### `face_analysis.py`
- Performs real-time face detection and emotion recognition.
- Utilizes OpenCV for video frame processing, Dlib for face detection, and a pre-trained Keras model for emotion recognition.
- Extracts facial landmarks to analyze features like smiles, cap wearing, and earrings.
- Predicts the emotion of the detected face.

### `train_emotion_model.py`
- Trains a deep learning model (CNN) for emotion recognition.
- Uses the FER2013 dataset for training and validation.
- Implements data augmentation techniques for better generalization.
- Saves the trained model for future use in emotion recognition.

## Usage

### Running Face Analysis
```
python face_analysis.py
```
- This script activates the webcam and starts the face analysis process.
- Press 'q' to quit the application.

### Training the Emotion Recognition Model
```
python train_emotion_model.py
```
- This will start the training process for the emotion recognition model.
- The trained model will be saved as `emotion_recognition_model.h5`.

## Author
- Jide Awoniyi
