import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import time
import tempfile

from extract import get_voice_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb

st.set_page_config(layout="wide")
st.title("🧠 Parkinson Detection (ML + DL Ensemble System)")

# =========================
# RECORD FUNCTION
# =========================
def record_audio_pro(duration=10, filename="recorded.wav", samplerate=22050):

    countdown = st.empty()
    for i in range(3, 0, -1):
        countdown.markdown(f"### ⏳ Starting in {i}...")
        time.sleep(1)

    countdown.empty()

    status = st.empty()
    status.info("🎤 Recording... Speak clearly")

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)

    progress = st.progress(0)

    for i in range(duration):
        time.sleep(1)
        progress.progress((i + 1) / duration)
        status.info(f"🎙 Recording {i+1}/{duration} sec")

    sd.wait()
    sf.write(filename, recording, samplerate)

    status.success("✅ Recording complete")
    return filename

# =========================
# AUDIO QUALITY
# =========================
def check_audio_quality(y):
    vol = np.mean(np.abs(y))
    if vol < 0.01:
        return "⚠️ Very low volume"
    elif vol < 0.03:
        return "⚠️ Low volume"
    return "✅ Good quality"

# =========================
# DATASET CREATION
# =========================
def create_dataset(folder="audio"):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            features = get_voice_features(path)

            if "healthy" in file.lower():
                features["status"] = 0
            elif "patient" in file.lower():
                features["status"] = 1
            else:
                continue

            features["filename"] = file
            data.append(features)

    df = pd.DataFrame(data)
    df.to_csv("voice_features.csv", index=False)
    return df

# =========================
# ML TRAINING
# =========================
def train_ml(df):

    X = df.drop(columns=["status", "filename"], errors="ignore")
    y = df["status"]

    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier()
    }

    best_model = None
    best_score = 0
    best_name = ""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc

        if acc > best_score:
            best_model = model
            best_score = acc
            best_name = name

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(best_name, "best_model_name.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "feature_columns.pkl")
    joblib.dump(results, "model_results.pkl")

    return best_name, best_score, results

# =========================
# DL TRAINING
# =========================
def train_dl(folder="audio"):

    X, y = [], []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        audio, sr = librosa.load(path, sr=22050)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        spec = librosa.power_to_db(spec)

        spec = np.resize(spec, (128,128))
        X.append(spec)

        y.append(0 if "healthy" in file else 1)

    X = np.array(X).reshape(-1,128,128,1)
    y = np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, verbose=0)

    model.save("cnn_model.h5")

# =========================
# TRAIN UI
# =========================
st.header("🚀 Training")

if st.button("Train Full System"):
    df = create_dataset("audio")

    best_name, best_score, results = train_ml(df)
    train_dl("audio")

    st.success(f"Best ML Model: {best_name} ({best_score*100:.2f}%)")

    # Accuracy Graph (Labeled)
    fig3, ax3 = plt.subplots()
    ax3.bar(results.keys(), results.values())
    ax3.set_title("Model Accuracy Comparison")
    ax3.set_xlabel("Models")
    ax3.set_ylabel("Accuracy")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# =========================
# LOAD MODELS
# =========================
if not os.path.exists("best_model.pkl"):
    st.warning("⚠️ Train model first")
    st.stop()

ml_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("feature_columns.pkl")
model_name = joblib.load("best_model_name.pkl")
results = joblib.load("model_results.pkl")
dl_model = tf.keras.models.load_model("cnn_model.h5")

# =========================
# MODEL INFO
# =========================
st.subheader("📊 Model Info")
st.info(f"Best Model Used: {model_name}")

# =========================
# INPUT
# =========================
st.header("🎤 Input")

option = st.radio("Choose Input Method", ["Record", "Upload"])
audio_path = None

if option == "Record":
    if st.button("Start Recording"):
        audio_path = record_audio_pro()
        st.audio(audio_path)

else:
    file = st.file_uploader("Upload WAV", type=["wav"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(file.read())
            audio_path = f.name
        st.audio(audio_path)

# =========================
# PREDICTION
# =========================
if audio_path:

    y, sr = librosa.load(audio_path, sr=22050)
    st.info(check_audio_quality(y))

    with st.spinner("Analyzing..."):

        features = get_voice_features(audio_path)
        df = pd.DataFrame([features])
        df = df.reindex(columns=columns, fill_value=0)

        ml_prob = ml_model.predict_proba(scaler.transform(df))[0][1]

        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec)
        spec = np.resize(spec, (128,128)).reshape(1,128,128,1)

        dl_prob = dl_model.predict(spec)[0][0]

        final_prob = (ml_prob + dl_prob) / 2

    label = "Parkinson's ❌" if final_prob > 0.6 else "Healthy ✅"

    st.subheader("🧾 Result")
    st.write(f"🤖 Model Used: **{model_name}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Score", f"{ml_prob*100:.2f}%")
    col2.metric("DL Score", f"{dl_prob*100:.2f}%")
    col3.metric("Final", label)

    st.progress(final_prob)

    # Waveform (Labeled)
    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(f"Waveform ({label})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    # Spectrogram (Labeled)
    with colB:
        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(
            spec[0,:,:,0],
            sr=sr,
            x_axis='time',
            y_axis='mel',
            cmap='magma',
            ax=ax2
        )
        ax2.set_title("Mel Spectrogram")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Frequency (Mel)")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)