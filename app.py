import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Load model (cache to avoid reloading every time)
@st.cache_resource
def load_my_model():
    return load_model("fake_voice_model.h5")

model = load_my_model()

# Feature extraction
def extract_mfcc(file, sr=16000, n_mfcc=40, max_len=87):
    y, _ = librosa.load(file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.reshape(1, 40, 87, 1)

# UI
st.title("🎙️ Fake Voice Detection App")
st.write("Upload an audio file (.wav, .mp3, .flac, .m4a)")

uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "m4a"])

if uploaded is not None:
    temp_path = "temp.wav"

    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    st.audio(temp_path)

    feat = extract_mfcc(temp_path)
    pred = model.predict(feat)[0][0]

    label = "🟢 REAL" if pred < 0.5 else "🔴 FAKE"
    confidence = 1 - pred if pred < 0.5 else pred

    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    os.remove(temp_path)  # cleanup