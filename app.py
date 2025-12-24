#app.py



import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from model import psnr # Import our custom metric

# --- Page Configuration ---
st.set_page_config(
    page_title="HD Image Super-Resolution",
    page_icon="⚫",
    layout="centered"
)

# --- Load The Trained Model ---
@st.cache_resource
def load_model():
    # Load the new 128x128 model
    model = tf.keras.models.load_model('sr_128_model.h5', custom_objects={'psnr': psnr})
    return model

model = load_model()

# --- UI Elements ---
st.title(" HD Image Super-Resolution")
st.write(
    "Upload any image. It will be downscaled to 64x64, then it will enhance it back to a sharp 128x128 image."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Processing Steps")

    # 1. Create the low-resolution version (64x64)
    lr_image = image.resize((64, 64), Image.BICUBIC)

    # 2. Create the model input by upscaling the LR image with bicubic interpolation (128x128)
    # This is the "before" image we compare against.
    bicubic_image = lr_image.resize((128, 128), Image.BICUBIC)

    # 3. Prepare the image for the model
    input_array = np.array(bicubic_image) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)

    # 4. Get AI prediction
    with st.spinner('Applying enhancement...'):
        predicted_tensor = model.predict(input_tensor)

    predicted_array = np.clip(predicted_tensor[0], 0.0, 1.0)

    # --- Calculate PSNR Metric ---
    psnr_value = tf.image.psnr(input_array, predicted_array, max_val=1.0).numpy()

    # --- Display Results ---
    st.subheader("Super Resoluted vs. Standard Upscaling")
    col1, col2 = st.columns(2)

 with col1:
    st.image(predicted_array, caption="Super-Resolved Output")

 with col2:
    st.image(bicubic_image, caption="Standard Bicubic Upscale")


    # --- Display Metric ---
    st.metric(label="PSNR (Output vs. Input)", value=f"{psnr_value:.2f} dB")
    st.info(
        "This PSNR score shows how much detail and clarity the model added compared to a standard resize. "
        "Higher is better!"

    )

