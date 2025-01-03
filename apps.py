import streamlit as st
import numpy as np
import librosa
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

@st.cache_resource  # Cache the loaded model for faster performance
def load_trained_model():
    return load_model('aktivitasmodel.h5')

# Define your label encoder (make sure the labels match your training data)
label_encoder = LabelEncoder()
label_encoder.fit(['Chew', 'Still', 'Drink', 'Run', 'Walk', 'Cough', 'Cry', 'Screaming', 'FallingObject', 'BrokenGlass'])

# Preprocessing function (same as before)
def preprocess_audio(path, sr=22050, augmented=False):
    y, sr_original = librosa.load(path, sr=sr)
    y = librosa.effects.preemphasis(y)
    yt, _ = librosa.effects.trim(y)
    yt = librosa.util.normalize(yt)
    return yt, sr

# Feature extraction function (you need to define this)
def extract_features(path, sr=22050, n_mfcc=13, hop_length=512):
    data, sr = preprocess_audio(path, sr=sr)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Streamlit app
st.title("Pengenalan Aktivitas Berdasarkan Suara")

# Load the model
model = load_trained_model()

# File uploader
uploaded_files = st.file_uploader("Pilih file suara", type=["wav"], accept_multiple_files=True)

WARNING_aCTIVITIES = ["FallingObject", "Cough", "Cry", "BrokenGlass"]
# Prediction button
if st.button("Prediksi"):
    if uploaded_files:
        audio_names = []
        predictions = []
        warnings = []  # List to collect warnings
        original_audio_data = []

        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(uploaded_file.read())
                temp_audio_file_path = temp_audio_file.name

            # Extract features and predict
            mfccs = extract_features(temp_audio_file_path)
            mfccs_reshaped = mfccs.reshape(1, mfccs.shape[0], 1)  # Reshape for model input
            y_pred = model.predict(mfccs_reshaped)
            y_pred_label = np.argmax(y_pred, axis=1)
            predicted_activity = label_encoder.inverse_transform(y_pred_label)[0]
            predictions.append(predicted_activity)

            # Check if the predicted activity is a warning activity
            if predicted_activity in WARNING_aCTIVITIES:
                warnings.append(f"⚠️ **WARNING:** Aktivitas '{predicted_activity}' memerlukan perhatian!")
            else:
                warnings.append("✅ Aktivitas Normal")

            # Delete temporary file
            os.remove(temp_audio_file_path)

        # Display results
        # Debugging log untuk prediksi dan warning
        for audio_name, prediction, warning in zip(uploaded_files, predictions, warnings):
            st.write(f"Audio File: {audio_name.name}")
            st.write(f"Prediksi Aktivitas: **{prediction}**")
            st.write(f"Warning Status: {warning}")  # Tambahkan log status warning
            
            # Display warning sign if necessary
            if "⚠️" in warning:  # Check if it's a warning
                st.warning(warning)
            else:
                st.success(warning)

        # Display results in a DataFrame
        result_df = pd.DataFrame({
            'File Audio': [file.name for file in uploaded_files],
            'Prediksi Aktivitas': predictions,
            'Status': warning
        })
        st.write("Hasil Prediksi dalam Tabel:")
        st.write(result_df)

    else:
        st.error("Silahkan unggah file suara terlebih dahulu.")
