import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------------------
# 1. Cargar el modelo (.keras)
# ------------------------------

@st.cache_resource
def load_model():
    ruta = "mi_modelo.keras"
    model = tf.keras.models.load_model(ruta)
    return model

model = load_model()

# ------------------------------
# 2. Clases
# ------------------------------
CLASSES = ["billetes", "Calculadora","llaves" ]

# ------------------------------
# 3. Preprocesamiento EfficientNetB0
# ------------------------------
def preprocess(img):
    img = img.resize((224, 224))  # tamaño estándar EfficientNetB0
    img = np.array(img)
    img = preprocess_input(img)   # normalización CORRECTA
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------
# 4. Interfaz Streamlit
# ------------------------------

st.title("Clasificador de objetos")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen cargada", width=300)

    # Procesar imagen
    input_img = preprocess(img)

    # Predicción
    preds = model.predict(input_img)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    # Mostrar resultados
    st.subheader("Resultado:")
    st.write(f"**Clase predicha:** {CLASSES[class_id]}")
    st.write(f"**Confianza:** {confidence:.2f}")



#cd "C:\Users\Usuario\OneDrive\Desktop\Universidad\Noveno semestre\IA"
#streamlit run pagweb.py

