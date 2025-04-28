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

st.set_page_config(layout="wide", initial_sidebar_state="auto")

# Load models
@st.cache_resource
def load_convnext_model():
    # Cek apakah model sudah ada
    if not os.path.exists('Model/ConvNeXtTiny1_model.keras'):
        # Kalau belum ada, download dari Google Drive
        url = 'https://drive.google.com/file/d/15cxTbXXeAf2OpoaPEjAiUnmp95AtCNT5/view?usp=sharing'  # Ganti YOUR_FILE_ID_CONVNEXT
        gdown.download(url, 'ConvNeXtTiny1_model.keras', quiet=False)
    model = tf.keras.models.load_model('ConvNeXtTiny1_model.keras')
    return model

@st.cache_resource
def load_inception_model():
    if not os.path.exists('Model/InceptionV31_model.keras'):
        url = 'https://drive.google.com/file/d/1GydV0gWsEofstdLm6cr0-fAOrl8lXkYp/view?usp=sharing'  # Ganti YOUR_FILE_ID_INCEPTION
        gdown.download(url, 'Model/InceptionV31_model.keras', quiet=False)
    model = tf.keras.models.load_model('InceptionV31_model.keras')
    return model
convnext_model = load_convnext_model()
inception_model = load_inception_model()

# Preprocessing functions
def preprocess_image_convnext(image: Image.Image):
    image = image.resize((512, 512))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    image_array = convnext_preprocess(image_array)
    return image_array

def preprocess_image_inception(image: Image.Image):
    image = image.resize((512, 512))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    image_array = inception_preprocess(image_array)
    return image_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.input, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients terhadap output feature map
    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def superimpose_heatmap(img, heatmap, alpha=0.4):
    import cv2

    # Convert PIL image to array
    img = np.array(img)

    # Resize heatmap ke ukuran gambar
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3] * 255
    heatmap = np.uint8(heatmap)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# Main page
def main_page():
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 450px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<h1 style='text-align: left; color: #4A90E2;'>🐛 MARIOAPP</h1>", unsafe_allow_html=True)
    st.sidebar.markdown(
            """
            <div style="text-align: left;">
            Aplikasi ini bertujuan untuk mengklasifikasikan tahapan instar dari <b>Crocidolomia pavonana</b> berdasarkan gambar yang Anda upload.
            Dengan menggunakan dua model deep learning yaitu <b>ConvNeXt Tiny</b> dan <b>Inception V3</b>, aplikasi ini memberikan hasil prediksi instar secara akurat.
            <br>Yuk, upload gambar ulatnya dan lihat hasil klasifikasinya!<br>
            <br><br>
            
            </div>
            """, unsafe_allow_html=True
        )

    st.sidebar.markdown("## Keunggulan Fitur")
    st.sidebar.markdown("✅ Menggunakan dua model deep learning")
    st.sidebar.markdown("✅ Memberikan hasil prediksi dan tingkat akurasi")
    st.sidebar.markdown("✅ Membantu dalam pengelompokan instar untuk pengendalian hama")
    
    # 📤 Upload gambar untuk prediksi
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Klasifikasi Tahapan Instar Crocidolomia Pavonana</h1>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader(label ="Upload gambar", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_column_width=True)

        if st.button("Klasifikasi Gambar"):
            with col2:
                status_placeholder = st.empty()  # Buat tempat kosong
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

                # st.markdown("### Hasil Klasifikasi")
                pred_col1, pred_col2 = st.columns(2)

                # Tampilkan hasil teks seperti biasa
                with pred_col1:
                    st.info(
                        f"**Model:** ConvNeXt Tiny\n\n"
                        f"**Prediksi:** {predicted_class_convnext}\n\n"
                        f"**Akurasi:** {confidence_convnext:.2f}%"
                    )

                with pred_col2:
                    st.info(
                        f"**Model:** Inception V3\n\n"
                        f"**Prediksi:** {predicted_class_inception}\n\n"
                        f"**Akurasi:** {confidence_inception:.2f}%"
                    )

                # ➡️ Prediksi sudah dalam variabel ini:
                # Setelah prediksi
                preprocessed_convnext = preprocess_image_convnext(image)
                prediction_convnext = convnext_model.predict(preprocessed_convnext)[0]  # [0] karena hasilnya (1,4)
                preprocessed_inception = preprocess_image_inception(image)
                prediction_inception = inception_model.predict(preprocessed_inception)[0]

                # Data untuk visualisasi
                class_names = ['Instar 1', 'Instar 2', 'Instar 3', 'Instar 4']

                # Buat DataFrame tabel
                df_confidence = pd.DataFrame({
                    'Tahap Instar': class_names,
                    'ConvNeXt Tiny (%)': prediction_convnext * 100,
                    'Inception V3 (%)': prediction_inception * 100
                })

                st.dataframe(df_confidence.style.format({'ConvNeXt Tiny (%)': '{:.2f}', 'Inception V3 (%)': '{:.2f}'}))
                    
            # Grad-CAM ConvNeXt Tiny
            gradcam_status_placeholder = st.empty()
            gradcam_status_placeholder.info("⏳ Membuat Grad-CAM visualisasi...")
            preprocessed_convnext = preprocess_image_convnext(image)
            preprocessed_inception = preprocess_image_inception(image)
            heatmap_convnext = make_gradcam_heatmap(preprocessed_convnext, convnext_model, "convnext_tiny_stage_3_block_2_identity")
            superimposed_img_convnext = superimpose_heatmap(image, heatmap_convnext)

            # Grad-CAM InceptionV3
            heatmap_inception = make_gradcam_heatmap(preprocessed_inception, inception_model, "mixed10")
            superimposed_img_inception = superimpose_heatmap(image, heatmap_inception)
            
            # Grad-CAM ConvNeXt Tiny
            heatmap_convnext = make_gradcam_heatmap(preprocessed_convnext, convnext_model, "convnext_tiny_stage_3_block_2_identity")
            superimposed_img_convnext = superimpose_heatmap(image, heatmap_convnext)

            # Grad-CAM InceptionV3
            heatmap_inception = make_gradcam_heatmap(preprocessed_inception, inception_model, "mixed10")
            superimposed_img_inception = superimpose_heatmap(image, heatmap_inception)

            gradcam_status_placeholder.success("✅ Grad-CAM berhasil dibuat!")

            # Tampilkan Grad-CAM
            st.markdown("### Grad-CAM Visualisasi")
            gradcam_col1, gradcam_col2 = st.columns(2)

            with gradcam_col1:
                st.image(superimposed_img_convnext, caption="Grad-CAM ConvNeXt Tiny", use_column_width=True)

            with gradcam_col2:
                st.image(superimposed_img_inception, caption="Grad-CAM InceptionV3", use_column_width=True)



    st.markdown("---")
    
    # ❗ Bagian tentang Crocidolomia pavonana
    with st.expander("Tentang Crocidolomia pavonana"):
        st.markdown("## Tentang Crocidolomia pavonana")
        
        croci_col1, croci_col2 = st.columns(2)

        with croci_col1:
            st.image("Gambar/crocidolomia_pavonana.jpg", caption="Crocidolomia pavonana", use_column_width=True)

        with croci_col2:
            st.markdown(
                """
                <div style="text-align: justify; font-size: 18px;">
                <b>Crocidolomia pavonana</b> adalah sejenis ulat yang menjadi hama utama pada tanaman sayuran dari famili Cruciferae seperti kubis, brokoli, dan kembang kol.
                Hama ini mengalami beberapa tahap pertumbuhan atau <b>instar</b>, yang setiap tahapnya memiliki ukuran dan karakteristik tubuh yang berbeda.
                Deteksi tahapan instar sangat penting untuk mengatur strategi pengendalian hama yang lebih efektif.
                </div>
                """, unsafe_allow_html=True
            )
    
    # 🖼️ Galeri tahapan instar
    with st.expander("Tahapan Instar Crocidolomia pavonana"):
        st.markdown("## Tahapan Instar Crocidolomia pavonana")

        instar_info = [
            {"title": "Instar 1", "image": "Gambar/instar1.jpg", "desc": "Instar 1 berukuran panjang tubuh 1,84–2,51 mm. Tubuh sangat kecil dan berwarna hijau pucat."},
            {"title": "Instar 2", "image": "Gambar/instar2.jpg", "desc": "Instar 2 berukuran panjang tubuh 5,1–6,82 mm. Mulai terlihat garis tubuh tipis."},
            {"title": "Instar 3", "image": "Gambar/instar3.jpg", "desc": "Instar 3 berukuran panjang tubuh 11,97–15,85 mm. Ukuran tubuh membesar dan warna lebih hijau."},
            {"title": "Instar 4", "image": "Gambar/instar4.jpg", "desc": "Instar 4 berukuran panjang tubuh 14,25–18,7 mm. Tubuh penuh, warna hijau tua, dan pola tubuh lebih jelas."}
        ]
        instar_cols = st.columns(len(instar_info))

        for idx, col in enumerate(instar_cols):
            with col:
                st.image(instar_info[idx]["image"], caption=instar_info[idx]["title"], use_column_width=True)
                st.caption(instar_info[idx]["desc"])

# Run app
if __name__ == "__main__":
    main_page()
