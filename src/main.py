import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Preparamos los datos

dir_dataset = "dataset-resized"

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

# Generamos los datos

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


