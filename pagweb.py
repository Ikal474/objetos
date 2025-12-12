import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


@st.cache_resource
def load_model():
    ruta = "mi_modelo.keras"
    model = tf.keras.models.load_model(ruta, compile=False)
    return model

model = load_model()

CLASSES = ["Calculadora", "billetes", "llaves"] 

def preprocess(img):
    
    if img.mode != "RGB":
        img = img.convert("RGB")    
    img = img.resize((224, 224))  
    img = np.array(img)           
    img = preprocess_input(img)  
    img = np.expand_dims(img, axis=0) 
    return img
    
st.title("Clasificador de objetos")
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

 if uploaded_file is not None:
    # 1. Cargar y mostrar imagen
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen cargada", width=300)

    # 2. Procesar imagen
    input_img = preprocess(img)
    # Usamos model(input, training=False) para asegurar la predicción determinista.
     raw_preds = model(input_img, training=False) 
    probs = raw_preds.numpy()[0] 
    class_id = np.argmax(probs)
    confidence = float(np.max(probs))        

    st.write(f"**Clase predicha:** **{CLASSES[class_id]}**")
        
    st.write(f"**Confianza:** `{confidence:.4f}`")

    st.text("Probabilidades por Clase:")
    for i, prob in enumerate(probs):
        st.text(f"- {CLASSES[i]}: {prob:.4f}")



#cd "C:\Users\Usuario\OneDrive\Desktop\Universidad\Noveno semestre\IA"
#streamlit run pagweb.py










