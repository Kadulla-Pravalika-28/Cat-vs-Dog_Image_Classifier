import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="🐱🐶 Cat vs Dog Classifier", layout="centered")
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image and let the model predict whether it’s a Cat 🐱 or a Dog 🐶.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_mobilenetv2.h5")

model = load_model()

uploaded = st.file_uploader("📸 Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    img = img.resize((160,160))
    arr = np.expand_dims(image.img_to_array(img), axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    pred = model.predict(arr)[0][0]
    if pred > 0.5:
        st.success(f"🐶 Dog ({pred*100:.2f}% confidence)")
    else:
        st.success(f"🐱 Cat ({(1-pred)*100:.2f}% confidence)")