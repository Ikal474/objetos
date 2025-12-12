import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import drive
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- PASO 1: Montar Google Drive ---
drive.mount('/content/drive')

# --- PASO 2: Configuración Inicial ---
# IMPORTANTE: Asegúrate de que esta ruta sea correcta y contenga subcarpetas por clase.
DIR_DATASET = '/content/drive/MyDrive/dataset_cosas' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Contenido de la carpeta de datos:")
try:
    print(os.listdir(DIR_DATASET))
except FileNotFoundError:
    print(f"ERROR: La ruta del dataset no existe: {DIR_DATASET}")
    
# --- PASO 3: Carga, División y Optimización de Datos ---
print("\nCargando conjuntos de datos...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)

# Mejor rendimiento (prefetch)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --- PASO 4: Definición de Aumentación de Datos ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# --- PASO 5: Carga y Congelación del Modelo Base (EfficientNetB0) ---
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)
# Congelar los pesos
base_model.trainable = False

# --- PASO 6: Ensamblaje de la Arquitectura del Modelo ---
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
# Usar training=False asegura que el modelo base actúe en modo inferencia
x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation="relu")(x)
# La capa de salida tiene tantas neuronas como clases detectadas
outputs = layers.Dense(len(class_names), activation="softmax")(x)

# Definición del modelo final
model = tf.keras.Model(inputs, outputs)

# --- PASO 7: Compilación del Modelo ---
print("\nCompilando el modelo...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# --- PASO 8: Entrenamiento del Modelo ---
print("\nIniciando entrenamiento...")
# Asegúrate de tener suficientes recursos (GPU)
hist = model.fit(train_ds, validation_data=val_ds, epochs=5) 

# --- PASO 9: Guardar el Modelo ---
# NOTA: Guardar en Drive para que sea persistente
MODEL_SAVE_PATH = "/content/drive/MyDrive/mi_modelo_final.keras"
model.save(MODEL_SAVE_PATH)
print(f"\nModelo guardado exitosamente en: {MODEL_SAVE_PATH}")

# --- PASO 10: Función de Preparación para la Inferencia ---
def preparar_imagen(ruta_o_archivo):
    # La ruta_o_archivo puede ser una ruta de disco o un objeto cargado (PIL Image)
    if isinstance(ruta_o_archivo, str):
        im = Image.open(ruta_o_archivo).convert("RGB")
    else: # Asume que es un objeto PIL Image ya cargado
        im = ruta_o_archivo.convert("RGB")

    im = im.resize(IMG_SIZE)
    arr = np.array(im)

    # Mostrar imagen solo en Colab
    if 'google.colab' in str(get_ipython()):
        plt.imshow(arr)
        plt.title("Imagen a clasificar")
        plt.axis("off")
        plt.show()

    x = np.expand_dims(arr, axis=0)
    x = preprocess_input(x)
    return x

# --- PASO 11: Prueba de Inferencia (Ejemplo) ---
# REEMPLAZA "key.jpeg" con la ruta a una imagen de prueba en tu Drive
# Si la imagen está en Drive, usa la ruta completa, ej: "/content/drive/MyDrive/imagenes_prueba/llave.jpg"
print("\n--- Prueba de Inferencia ---")
try:
    x_test = preparar_imagen("key.jpeg") # Ajusta la ruta si es necesario
    
    # La clave para la inferencia determinista es usar model() con training=False
    pred_tensor = model(x_test, training=False) 
    
    # Convertir a numpy
    pred = pred_tensor.numpy()[0] 
    idx = np.argmax(pred)
    
    print("Probabilidades:", pred)
    print(f"Predicción: {class_names[idx]} (Confianza: {np.max(pred):.4f})")
except FileNotFoundError:
    print("ERROR: No se encontró la imagen de prueba 'key.jpeg'.")



#cd "C:\Users\Usuario\OneDrive\Desktop\Universidad\Noveno semestre\IA"
#streamlit run pagweb.py







