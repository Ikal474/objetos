import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# Borramos la importación innecesaria para evitar confusión
# from tensorflow.keras.applications.efficientnet import preprocess_input 

@st.cache_resource
def load_model():
    ruta = "mi_modelo.keras"
    model = tf.keras.models.load_model(ruta, compile=False)
    return model

model = load_model()

CLASSES = ["PS4", "billetes", "llaves"] 

def preprocess(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 1. Redimensionar
    img = img.resize((224, 224))
    
    # 2. Convertir a Array (quedan valores de 0 a 255)
    img = np.array(img)
    
    # --- ELIMINAMOS ESTA LÍNEA ---
    # img = preprocess_input(img) 
    # -----------------------------
    
    # 3. Expandir dimensiones para crear el lote (batch) de 1 imagen
    # El modelo espera (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img
    
st.title("Clasificador de objetos")
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen cargada", width=300)

    input_img = preprocess(img)
    
    try:
        raw_preds = model(input_img, training=False)
        probs = raw_preds.numpy()[0]
        class_id = np.argmax(probs)
        confidence = float(np.max(probs))

        st.write(f"**Clase predicha:** **{CLASSES[class_id]}**")
        st.write(f"**Confianza:** `{confidence:.4f}`")
        
        # Opcional: Mostrar todas las probabilidades para depurar
        # st.write(f"Probabilidades: {probs}")
        
    except Exception as e:
        st.error(f"Hubo un error al procesar la imagen: {e}")


#cd "C:\Users\Usuario\OneDrive\Desktop\Universidad\Noveno semestre\IA"
#streamlit run pagweb.py

















