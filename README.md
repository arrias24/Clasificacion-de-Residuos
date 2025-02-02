# Clasificador de Residuos

Este proyecto es un clasificador de residuos basado en una red neuronal convolucional (CNN) utilizando TensorFlow y Keras. El modelo clasifica imágenes de residuos en seis categorías: Cartón, Vidrio, Metal, Papel, Plástico y Otros. Además, cuenta con una interfaz gráfica simple construida con Tkinter para facilitar la carga y clasificación de imágenes.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalados los siguientes paquetes:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Pillow
- scikit-learn
- Tkinter (generalmente incluido en instalaciones de Python)

Puedes instalar las dependencias necesarias utilizando pip:

## Estructura del Proyecto

Asegúrate de que la estructura del proyecto sea la siguiente:

/tu-proyecto
│
├── dataset-resized/ # Directorio que contiene las imágenes organizadas en carpetas por clase
│ ├── Carton/
│ ├── Vidrio/
│ ├── Metal/
│ ├── Papel/
│ ├── Plastico/
│ └── Otros/
│
├── modelo_entrenado.h5 # Archivo del modelo entrenado (se generará automáticamente si no existe)
└── main.py # Archivo principal del proyecto (el código proporcionado)


## Instrucciones para Ejecutar el Proyecto

1. **Prepara tu Dataset**:
   - Organiza tus imágenes en carpetas dentro del directorio `dataset-resized`, donde cada carpeta debe llevar el nombre de la clase correspondiente.

2. **Ejecuta el Script**:
   - Abre tu terminal o línea de comandos.
   - Navega al directorio donde se encuentra tu archivo `main.py`.
   - Ejecuta el siguiente comando:

3. **Interfaz Gráfica**:
- Se abrirá una ventana con la interfaz gráfica.
- Haz clic en "Cargar Imagen" para seleccionar una imagen desde tu computadora.
- El clasificador mostrará el tipo de residuo detectado.

4. **Cierre de la Aplicación**:
- Para cerrar la aplicación, haz clic en "Salir".

## Evaluación del Modelo

El modelo se evalúa automáticamente utilizando un conjunto de validación al inicio. Se generará una matriz de confusión y un reporte de clasificación que se mostrarán en la consola.

## Notas

- El modelo se entrena durante 3 épocas y se guarda como `modelo_entrenado.h5`. Si ya existe este archivo, el modelo se cargará en lugar de ser entrenado nuevamente.
- Puedes ajustar los parámetros del modelo y el número de épocas en el código según sea necesario.
