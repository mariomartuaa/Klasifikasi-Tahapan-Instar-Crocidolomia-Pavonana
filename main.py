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

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .main .block-container {
            padding-top: 0rem;
            height: 500%;
        }
        
[data-testid="stHeader"] {
    background-color: #ffff;
}

[data-testid="stAppViewBlockContainer"] {
    background: linear-gradient(130deg, #fdf6ec 0%, #e6f4ea 50%, #fff9c4 100%);
}

[data-testid="baseButton-headerNoPadding"], [data-testid="baseButton-minimal"], [data-testid="stUploadedFile"], [data-testid="stFileUploadDropzone"], [data-testid="stFileDropzoneInstructions"] {
    color:#2e5339;
}

.st-emotion-cache-7oyrr6 e1bju1570 {
    color:#2e5339;
}

[data-testid="baseButton-secondary"] {
    background-color: white;
}

[data-testid="stSidebarUserContent"], [data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #e9f5db 0%, #c7e9b0 40%, #fef9c3 100%); 
}

[data-testid="stSidebarUserContent"]{
    height: 100%; 
    padding-top: 2rem;
}

.banner {
    background-image: url('https://png.pngtree.com/thumb_back/fh260/background/20230912/pngtree-the-whole-field-was-full-of-cabbages-image_13120953.png');
    background-size: cover;
    background-position: center;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
.banner h1 {
    font-size: 3rem;
    color: #ffffff;
    margin: 10px;
    text-shadow: 1px 1px 3px #000;
}
.banner h2 {
    font-size: 1.5rem;
    color: #f0f0f0;
    margin: 0 20px;
    text-shadow: 1px 1px 2px #000;
}
.banner button {
    height: 45px;
    padding: 0 40px;
    border-radius: 100px;
    border: 1px solid #ffffff;
    background-color: rgba(255, 255, 255, 0.8);
    color: #1b4332;
    font-size: 16px;
    cursor: pointer;
    margin-top: 20px;
}

.streamlit-expanderHeader {
    color: red;
}

[data-testid="stMarkdownContainer"]{
    color:#2e5339;
}

/* Judul besar */
.big-title {
    font-size: 36px;
    font-weight: 700;
    color: #1b4332;
}

/* Subjudul */
.sub-title {
    font-size: 20px;
    color: #3a5a40;
}

/* Kartu fitur dan deskripsi instar */
.card {
    border-radius: 10px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    background-color: #ffff;
    border: 1px solid #2e5339;
    color: #2e5339;
}

.card-informasi {
    border-radius: 10px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    background-color: #ffff;
    border: 1px solid #2e5339;
    color: #2e5339;
    height: 250px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_inception_model():
    model_path = 'InceptionV31_model.keras'
    if not os.path.exists(model_path):
        url = 'https://drive.google.com/uc?id=1GydV0gWsEofstdLm6cr0-fAOrl8lXkYp'
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

inception_model = load_inception_model()

# Preprocessing functions
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
    st.sidebar.markdown("<h1 style='text-align: center; color: #2e5339;'>Langkah-Langkah Penggunaan</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    1. **Upload Gambar**  
    Unggah gambar larva Crocidolomia Pavonana berformat `.jpg`, `.jpeg`, atau `.png`.

    2. **Lihat Pratinjau Gambar**  
    Pratinjau gambar akan otomatis muncul di halaman utama.

    3. **Klik 'Klasifikasi Gambar'**  
    Untuk memulai prediksi tahap instar dengan model deep learning.

    4. **Tunggu Proses Prediksi**  
    Sistem akan menampilkan hasil klasifikasi dan tingkat akurasi.

    5. **Tinjau Hasil**  
    Hasil berupa kelas instar, akurasi, dan tabel confidence.

    6. **Lihat Visualisasi Grad-CAM**  
    Menampilkan area penting dari gambar berdasarkan prediksi model.
    """)

    # 📤 Upload gambar untuk prediksi
    st.markdown("""<h1 style="text-align: center; font-size: 40px; color: #2e5339;">Klasifikasi Tahapan Instar Crocidolomia Pavonana</h1>""", unsafe_allow_html=True)
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

                # Prediksi InceptionV3
                preprocessed_inception = preprocess_image_inception(image)
                prediction_inception = inception_model.predict(preprocessed_inception)
                predicted_class_inception = class_names[np.argmax(prediction_inception)]
                confidence_inception = np.max(prediction_inception) * 100

                status_placeholder.success("✅ Klasifikasi selesai!")
                st.markdown(f"""
                    <div class="card">
                        <strong>Model: </strong>InceptionV3<br>
                        <strong>Prediksi: </strong>{predicted_class_inception}<br>
                        <strong>Akurasi: </strong>{confidence_inception:.2f}%<br>
                    </div>
                                    """, unsafe_allow_html=True)
                

                # Data untuk visualisasi
                df_confidence = pd.DataFrame({
                    'Tahap Instar': class_names,
                    'Akurasi (%)': prediction_inception[0] * 100
                })

                st.dataframe(df_confidence.style.format({'Akurasi (%)': '{:.2f}'}))

            gradcam_status_placeholder = st.empty()
            gradcam_status_placeholder.info("⏳ Membuat Grad-CAM visualisasi...")

            # Grad-CAM InceptionV3
            heatmap_inception = make_gradcam_heatmap(preprocessed_inception, inception_model, "mixed10")
            heatmap_inception = heatmap_inception.numpy()
            superimposed_img_inception = superimpose_heatmap(image, heatmap_inception)

            gradcam_status_placeholder.success("✅ Grad-CAM berhasil dibuat!")

            # Tampilkan Grad-CAM
            st.markdown(f'<h1 style="text-align: center; font-size: 30px; color: #2e5339;">Grad-CAM Visualisasi</h1>', unsafe_allow_html=True)
            gradcam_col1, gradcam_col2, gradcam_col3 = st.columns(3)
            with gradcam_col1:
                st.write("")
            with gradcam_col2:
                st.image(superimposed_img_inception, caption="Grad-CAM InceptionV3", use_column_width=True)
            with gradcam_col3:
                st.write("")


# Run app
if __name__ == "__main__":
    main_page()
