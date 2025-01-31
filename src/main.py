import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

dir_dataset = "dataset-resized"

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dir_dataset,
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dir_dataset,
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(100, activation='relu'), 
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
])

