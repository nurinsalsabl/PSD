import streamlit as st
import numpy as np
import librosa
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

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

# Prediction button
if st.button("Prediksi"):
    if uploaded_files:
        audio_names = []
        predictions = []
        original_audio_data = []

        for uploaded_file in uploaded_files:
            # Get audio data and filename
            audio_bytes = uploaded_file.read()  # Read audio bytes
            audio_name = uploaded_file.name
            audio_names.append(audio_name)

            # Save audio bytes temporarily for librosa to load
            with open(audio_name, "wb") as f:
                f.write(audio_bytes)
            
            # Extract features and predict
            mfccs = extract_features(audio_name)
            mfccs_reshaped = mfccs.reshape(1, mfccs.shape[0], 1) 
            y_pred = model.predict(mfccs_reshaped)
            y_pred_label = np.argmax(y_pred, axis=1)
            predicted_activity = label_encoder.inverse_transform(y_pred_label)[0]
            predictions.append(predicted_activity)

            # Get original audio for playback in Streamlit
            original_audio, sr = preprocess_audio(audio_name)
            original_audio_data.append((original_audio, sr))

        # Display results
        st.write("Hasil Prediksi:")
        for i, (audio_name, prediction) in enumerate(zip(audio_names, predictions)):
            st.write(f"Audio File: {audio_name}")
            st.audio(original_audio_data[i][0], format='audio/wav', sample_rate=original_audio_data[i][1], start_time=0)
            st.write(f"Prediksi Aktivitas: {prediction}")
            st.write("")

        # Display results in a DataFrame
        result_df = pd.DataFrame({
            'File Audio': audio_names,
            'Prediksi Aktivitas': predictions
        })
        st.write("Hasil Prediksi:")
        st.write(result_df)

    else:
        st.error("Silakan unggah file suara terlebih dahulu.")