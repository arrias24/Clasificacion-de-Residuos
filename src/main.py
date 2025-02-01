import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directorio del dataset
dir_dataset = "dataset-resized"
model_path = "modelo_entrenado.h5" 

# Generador de datos con aumento y división entre entrenamiento y validación

datagen = ImageDataGenerator( rescale=1.0/255.0, validation_split=0.2 )

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

# Verificar si el modelo ya existe

if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model(model_path)
else:
    # Modelo base MobileNetV2 preentrenado en ImageNet

    base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Construcción del modelo final
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(100, activation='relu'), 
        layers.Dropout(0.3),
        layers.Dense(6, activation='softmax')  # 6 clases de residuos
    ])

    # Compilación del modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(train_generator, validation_data=val_generator, epochs=3)

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

# Nombres de las clases (deben coincidir con las carpetas del dataset)
class_names = ["Carton", "Vidrio", "Metal", "Papel", "Plastico", "Otros"]

# Evaluación del modelo en el conjunto de validación
print("Evaluando el modelo...")
y_true = val_generator.classes  # Etiquetas reales de las imágenes de validación
y_pred = model.predict(val_generator)  # Predicciones del modelo para el conjunto de validación
y_pred_classes = np.argmax(y_pred, axis=1)  # Convertir probabilidades a clases predichas

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación (precisión, recall y F1-score)
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print("Reporte de Clasificación:")
print(report)

# Función para predecir una imagen individual (sin cambios)
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para el modelo
    predictions = model.predict(img_array)
    return np.argmax(predictions[0])  # Retorna la clase predicha

# Función para cargar y clasificar una imagen desde la interfaz gráfica (Tkinter) (sin cambios)
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        class_index = predict_image(file_path)
        result_label.config(text=f'Tipo de Residuo: {class_names[class_index]}')

# Función para cerrar la aplicación Tkinter (sin cambios)
def close_app():
    root.quit()

# Configuración de la ventana principal (Interfaz Gráfica con Tkinter) (sin cambios)
root = tk.Tk()
root.title("Clasificador de Residuos")
root.geometry("400x500")
root.configure(bg="#f0f0f0")

instructions_label = tk.Label(root, text="Selecciona una imagen para clasificar:", bg="#f0f0f0", font=("Arial", 12))
instructions_label.pack(pady=10)

panel = tk.Label(root, bg="#ffffff")
panel.pack(pady=20)

load_button = tk.Button(root, text="Cargar Imagen", command=load_image, bg="#4CAF50", fg="white", font=("Arial", 12))
load_button.pack(pady=10)

result_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 14))
result_label.pack(pady=20)

exit_button = tk.Button(root, text="Salir", command=close_app, bg="#f44336", fg="white", font=("Arial", 12))
exit_button.pack(pady=10)

root.mainloop()
