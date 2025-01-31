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

