#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 02:10:56 2023

@author: Smartsys
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

initial_learning_rate = 0.0001
decay_rate = 1e-6
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=decay_rate, staircase=True)
optimizer = Adam(learning_rate=lr_schedule)



# Load the FER2013 dataset (assuming it's in the same directory as your script)
data = pd.read_csv("fer2013.csv")

# Pre-process the data
pixels = data["pixels"].apply(lambda x: np.fromstring(x, sep=" ").reshape(48, 48, 1))
pixels = np.stack(pixels, axis=0)
emotions = to_categorical(data["emotion"])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(pixels, emotions, test_size=0.1, random_state=42)

# Define the VGG16 model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),

    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])


# Create the legacy Adam optimizer with the desired learning rate and decay
optimizer = Adam(learning_rate=0.0001, decay=1e-6)


# Create the ExponentialDecay learning rate schedule
initial_learning_rate = 0.0001
decay_rate = 1e-6
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=decay_rate, staircase=True)
optimizer = Adam(learning_rate=lr_schedule)


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          steps_per_epoch=len(X_train) / 64,
          epochs=30,
          verbose=1,
          validation_data=(X_val, y_val),
          shuffle=True)

# Save the model
model.save("emotion_recognition_model.h5")
