
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from model import psnr

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="HD Image Super-Resolution",
    page_icon="⚫",
    layout="centered"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "sr_128_model.h5",
        custom_objects={"psnr": psnr}
    )
    return model


model = load_model()

# ---------------- UI ----------------
st.title("HD Image Super-Resolution")
st.write(
    "Upload any image. It will be downscaled to 64x64, then enhanced to 128x128."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Main Logic ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Processing Steps")

    # 1. Downscale to 64x64
    lr_image = image.resize((64, 64), Image.BICUBIC)

    # 2. Upscale to 128x128 using bicubic
    bicubic_image = lr_image.resize((128, 128), Image.BICUBIC)

    # 3. Prepare input for model
    input_array = np.array(bicubic_image) / 255.0
    input_tensor = np.expand_dims(input_array, axis=0)

    # 4. Predict
    with st.spinner("Applying enhancement..."):
        predicted_tensor = model.predict(input_tensor)

    predicted_array = np.clip(predicted_tensor[0], 0.0, 1.0)

    # 5. PSNR calculation
    psnr_value = tf.image.psnr(
        input_array,
        predicted_array,
        max_val=1.0
    ).numpy()

    # ---------------- Display Results ----------------
    st.subheader("Super Resolved vs Standard Upscaling")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            predicted_array,
            caption="Super-Resolved Output",
            use_container_width=True
        )

    with col2:
        st.image(
            bicubic_image,
            caption="Standard Bicubic Upscale",
            use_container_width=True
        )

    st.metric(
        label="PSNR (Output vs Input)",
        value=f"{psnr_value:.2f} dB"
    )

    st.info(
        "Higher PSNR means better image quality compared to standard upscaling."
    )
