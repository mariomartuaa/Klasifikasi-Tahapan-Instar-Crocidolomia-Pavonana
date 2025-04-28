import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import os
import gdown
import cv2

st.set_page_config(layout="wide", initial_sidebar_state="auto")

# Load models
@st.cache_resource
def load_convnext_model():
    model_path = 'ConvNeXtTiny1_model.keras'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=15cxTbXXeAf2OpoaPEjAiUnmp95AtCNT5'
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_inception_model():
    model_path = 'InceptionV31_model.keras'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1GydV0gWsEofstdLm6cr0-fAOrl8lXkYp'
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

convnext_model = load_convnext_model()
inception_model = load_inception_model()

# Preprocessing functions
def preprocess_image_convnext(image: Image.Image):
    image = image.resize((512, 512))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return convnext_preprocess(image_array)

def preprocess_image_inception(image: Image.Image):
    image = image.resize((512, 512))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return inception_preprocess(image_array)

# Grad-CAM functions
@tf.function(reduce_retracing=True)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(model.input, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

def superimpose_heatmap(img, heatmap, alpha=0.4):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3] * 255
    heatmap = np.uint8(heatmap)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# Main page
def main_page():
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 450px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<h1 style='text-align: left; color: #4A90E2;'>🐛 MARIOAPP</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    Aplikasi ini bertujuan untuk mengklasifikasikan tahapan instar dari <b>Crocidolomia pavonana</b> berdasarkan gambar yang Anda upload.
    Dengan menggunakan dua model deep learning yaitu <b>ConvNeXt Tiny</b> dan <b>Inception V3</b>, aplikasi ini memberikan hasil prediksi instar secara akurat.
    Yuk, upload gambar ulatnya dan lihat hasil klasifikasinya!<br><br>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## Keunggulan Fitur")
    st.sidebar.markdown("✅ Menggunakan dua model deep learning")
    st.sidebar.markdown("✅ Memberikan hasil prediksi dan tingkat akurasi")
    st.sidebar.markdown("✅ Membantu dalam pengelompokan instar untuk pengendalian hama")

    # 📤 Upload gambar untuk prediksi
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Klasifikasi Tahapan Instar Crocidolomia Pavonana</h1>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader(label="Upload gambar", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, use_column_width=True)

        if st.button("Klasifikasi Gambar"):
            with col2:
                status_placeholder = st.empty()
                status_placeholder.info("⏳ Memproses dan memprediksi gambar...")

                # Mapping kelas
                class_names = ['Instar 1', 'Instar 2', 'Instar 3', 'Instar 4']

                # Prediksi ConvNeXt
                preprocessed_convnext = preprocess_image_convnext(image)
                prediction_convnext = convnext_model.predict(preprocessed_convnext)
                predicted_class_convnext = class_names[np.argmax(prediction_convnext)]
                confidence_convnext = np.max(prediction_convnext) * 100

                # Prediksi InceptionV3
                preprocessed_inception = preprocess_image_inception(image)
                prediction_inception = inception_model.predict(preprocessed_inception)
                predicted_class_inception = class_names[np.argmax(prediction_inception)]
                confidence_inception = np.max(prediction_inception) * 100

                status_placeholder.success("✅ Klasifikasi selesai!")

                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.info(f"**Model:** ConvNeXt Tiny\n\n**Prediksi:** {predicted_class_convnext}\n\n**Akurasi:** {confidence_convnext:.2f}%")
                with pred_col2:
                    st.info(f"**Model:** Inception V3\n\n**Prediksi:** {predicted_class_inception}\n\n**Akurasi:** {confidence_inception:.2f}%")

                # Data untuk visualisasi
                df_confidence = pd.DataFrame({
                    'Tahap Instar': class_names,
                    'ConvNeXt Tiny (%)': (prediction_convnext[0] * 100),
                    'Inception V3 (%)': (prediction_inception[0] * 100)
                })


                st.dataframe(df_confidence.style.format({'ConvNeXt Tiny (%)': '{:.2f}', 'Inception V3 (%)': '{:.2f}'}))

            # Grad-CAM ConvNeXt Tiny
            gradcam_status_placeholder = st.empty()
            gradcam_status_placeholder.info("⏳ Membuat Grad-CAM visualisasi...")

            heatmap_convnext = make_gradcam_heatmap(preprocessed_convnext, convnext_model, "convnext_tiny_stage_3_block_2_identity")
            heatmap_convnext = heatmap_convnext.numpy()
            superimposed_img_convnext = superimpose_heatmap(image, heatmap_convnext)

            # Grad-CAM InceptionV3
            heatmap_inception = make_gradcam_heatmap(preprocessed_inception, inception_model, "mixed10")
            heatmap_inception = heatmap_inception.numpy()
            superimposed_img_inception = superimpose_heatmap(image, heatmap_inception)

            gradcam_status_placeholder.success("✅ Grad-CAM berhasil dibuat!")

            # Tampilkan Grad-CAM
            st.markdown("### Grad-CAM Visualisasi")
            gradcam_col1, gradcam_col2 = st.columns(2)
            with gradcam_col1:
                st.image(superimposed_img_convnext, caption="Grad-CAM ConvNeXt Tiny", use_column_width=True)
            with gradcam_col2:
                st.image(superimposed_img_inception, caption="Grad-CAM InceptionV3", use_column_width=True)

# Run app
if __name__ == "__main__":
    main_page()
