import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. Cargar el modelo (.keras) ---
@st.cache_resource
def load_model():
    # ASEGÚRATE DE QUE "mi_modelo.keras" ESTÉ EN LA MISMA CARPETA DE TU SCRIPT
    ruta = "mi_modelo.keras" 
    try:
        # Usamos compile=False para evitar problemas de compatibilidad de optimizadores
        # si no vas a seguir entrenando el modelo en Streamlit.
        model = tf.keras.models.load_model(ruta, compile=False) 
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: Asegúrate de que '{ruta}' existe en el directorio de la aplicación y que las dependencias (TensorFlow/Keras) son correctas. Error: {e}")
        return None

model = load_model()

# --- 2. Clases ---
# El orden debe coincidir con el orden que Keras usó en el entrenamiento.
CLASSES = ["Calculadora", "billetes", "llaves"] 

# --- 3. Preprocesamiento EfficientNetB0 (CORREGIDO) ---
def preprocess(img):
    # img es un objeto PIL Image
    
    # CRÍTICO 1: Asegurar que la imagen sea RGB (3 canales), ya que EfficientNet lo espera.
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img = img.resize((224, 224))  # Redimensionamiento (224x224)
    img = np.array(img)           # Conversión a NumPy
    img = preprocess_input(img)   # Normalización específica de EfficientNet
    
    # Añadir la dimensión del lote: (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0) 
    return img

# --- 4. Interfaz Streamlit y Predicción ---
st.title("Clasificador de objetos 🧠")
st.markdown(f"Modelo entrenado para clasificar: **{', '.join(CLASSES)}**")

if model is not None:
    uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Cargar y mostrar imagen
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", width=300)

        # 2. Procesar imagen
        input_img = preprocess(img)

        # 3. Predicción (CORRECCIÓN CRÍTICA: Desactivar Aumentación de Datos)
        # Usamos model(input, training=False) para asegurar la predicción determinista.
        raw_preds = model(input_img, training=False) 
        
        # Convertir a NumPy para obtener las probabilidades
        probs = raw_preds.numpy()[0] 
        class_id = np.argmax(probs)
        confidence = float(np.max(probs))

        # 4. Mostrar resultados
        st.subheader("Resultado de la Clasificación:")
        
        # Muestra la clase predicha
        st.write(f"**Clase predicha:** **{CLASSES[class_id]}**")
        
        # Muestra la confianza
        st.write(f"**Confianza:** `{confidence:.4f}`")

        # 5. Desglose de Probabilidades (Debugging opcional)
        st.markdown("---")
        st.text("Probabilidades por Clase:")
        for i, prob in enumerate(probs):
            st.text(f"- {CLASSES[i]}: {prob:.4f}")



#cd "C:\Users\Usuario\OneDrive\Desktop\Universidad\Noveno semestre\IA"
#streamlit run pagweb.py








