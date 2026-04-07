import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
import noisereduce as nr
import tensorflow as tf

from extract import get_voice_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams["figure.figsize"] = (5,3)

st.set_page_config(layout="wide")
st.title("🧠 Parkinson Detection (ML + CNN + LSTM Hybrid)")

# =========================
# AUDIO LOAD
# =========================
def safe_load_audio(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            return None, None
        return librosa.load(path, sr=22050)
    except:
        return None, None

# =========================
# CLEAN AUDIO
# =========================
def clean_audio(y, sr):
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except:
        return y

# =========================
# AUGMENTATION
# =========================
def augment_audio(y, sr):
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    return [y, y_noise, y_pitch]

# =========================
# SPECTROGRAM SEQUENCE
# =========================
def extract_spectrogram_sequence(y, sr):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec = librosa.power_to_db(spec)
    spec = spec.T

    max_len = 128
    if spec.shape[0] < max_len:
        spec = np.pad(spec, ((0, max_len - spec.shape[0]), (0, 0)))
    else:
        spec = spec[:max_len]

    # Normalize
    spec = (spec - np.mean(spec)) / np.std(spec)

    return spec

# =========================
# CNN + LSTM MODEL
# =========================
def build_cnn_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,128)),
        tf.keras.layers.Reshape((128,128,1)),

        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Reshape((30, -1)),

        tf.keras.layers.LSTM(64),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================
# ML TRAIN
# =========================
def train_ml(X_train, X_test, y_train, y_test):
    models = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    best_model, best_score = None, 0
    for m in models.values():
        m.fit(X_train, y_train)
        score = m.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = m

    return best_model

# =========================
# TRAIN ALL
# =========================
def train_all(progress):

    # ===== ML =====
    df = []
    for file in os.listdir("audio"):
        if file.endswith(".wav"):
            path = os.path.join("audio", file)
            y, sr = safe_load_audio(path)
            if y is None: continue

            features = get_voice_features(path)

            if "healthy" in file.lower():
                features["status"] = 0
            elif "patient" in file.lower():
                features["status"] = 1
            else:
                continue

            df.append(features)

    df = pd.DataFrame(df)

    X = df.drop("status", axis=1).fillna(0)
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ml_model = train_ml(X_train, X_test, y_train, y_test)

    joblib.dump(ml_model, "ml.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "cols.pkl")

    # ===== DL =====
    progress.progress(50, text="Training DL...")

    X_dl, y_dl = [], []

    for file in os.listdir("audio"):
        if file.endswith(".wav"):
            path = os.path.join("audio", file)
            y_audio, sr = safe_load_audio(path)
            if y_audio is None: continue

            for aug in augment_audio(y_audio, sr):
                spec = extract_spectrogram_sequence(aug, sr)
                X_dl.append(spec)

                if "healthy" in file.lower():
                    y_dl.append(0)
                else:
                    y_dl.append(1)

    X_dl = np.array(X_dl)
    y_dl = np.array(y_dl)

    model = build_cnn_lstm()

    history = model.fit(
        X_dl, y_dl,
        epochs=10,
        batch_size=8,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
        verbose=0
    )

    model.save("dl_model.h5")
    joblib.dump(history.history, "history.pkl")

    progress.progress(100, text="Done")

# =========================
# UI
# =========================
st.header("🚀 Training")
if st.button("Train"):
    progress = st.progress(0)
    train_all(progress)

# =========================
# LOAD
# =========================
if not os.path.exists("ml.pkl"):
    st.stop()

ml_model = joblib.load("ml.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("cols.pkl")

dl_model = None
if os.path.exists("dl_model.h5"):
    dl_model = tf.keras.models.load_model("dl_model.h5")

# =========================
# PREDICTION
# =========================
file = st.file_uploader("Upload WAV", type=["wav"])

if file:
    path = tempfile.mktemp(suffix=".wav")
    with open(path, "wb") as f:
        f.write(file.read())

    y, sr = safe_load_audio(path)

    # ML
    feat = get_voice_features(path)
    df = pd.DataFrame([feat]).reindex(columns=cols, fill_value=0)
    ml_prob = ml_model.predict_proba(scaler.transform(df))[0][1]

    # DL
    spec = extract_spectrogram_sequence(y, sr)
    spec = np.expand_dims(spec, axis=0)
    dl_prob = dl_model.predict(spec)[0][0]

    # Final
    final_prob = 0.4*ml_prob + 0.6*dl_prob

    st.subheader("Results")
    st.write("ML:", ml_prob)
    st.write("DL:", dl_prob)
    st.write("Final:", final_prob)

# =========================
# TRAINING GRAPH
# =========================
if os.path.exists("history.pkl"):
    hist = joblib.load("history.pkl")

    st.subheader("📈 Training Performance")

    fig, ax = plt.subplots()
    ax.plot(hist["loss"], label="Loss")
    ax.plot(hist["val_loss"], label="Val Loss")
    ax.set_title("Training Loss")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(hist["accuracy"], label="Accuracy")
    ax.plot(hist["val_accuracy"], label="Val Accuracy")
    ax.set_title("Training Accuracy")
    ax.legend()
    st.pyplot(fig)
