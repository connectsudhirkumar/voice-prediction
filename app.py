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

from extract import get_voice_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams["figure.figsize"] = (5,3)

st.set_page_config(layout="wide")
st.title("🧠 Parkinson Detection (ML System)")

# =========================
# SAFE AUDIO LOAD
# =========================
def safe_load_audio(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            return None, None
        y, sr = librosa.load(path, sr=22050)
        if y is None or len(y) == 0:
            return None, None
        return y, sr
    except:
        return None, None


# =========================
# NOISE REDUCTION
# =========================
def clean_audio(y, sr):
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except:
        return y


# =========================
# DATASET
# =========================
def create_dataset(folder="audio"):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            y, sr = safe_load_audio(path)
            if y is None:
                continue

            y = clean_audio(y, sr)

            try:
                features = get_voice_features(path)
            except:
                continue

            if "healthy" in file.lower():
                features["status"] = 0
            elif "patient" in file.lower():
                features["status"] = 1
            else:
                continue

            data.append(features)

    return pd.DataFrame(data)


# =========================
# TRAIN ML MODELS
# =========================
def train_ml_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    best_model, best_name, best_score = None, "", 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(best_name, "best_model_name.pkl")

    return best_model, best_name


# =========================
# TRAIN ALL
# =========================
def train_all(progress):

    progress.progress(10, text="📂 Creating dataset...")
    df = create_dataset()

    if df.empty:
        st.error("❌ No valid audio files")
        return

    X = df.drop("status", axis=1)
    y = df["status"]

    progress.progress(30, text="🔧 Preparing data...")
    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X.columns.tolist(), "cols.pkl")

    progress.progress(50, text="🤖 Training ML models...")
    model, model_name = train_ml_models(X_train, X_test, y_train, y_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    joblib.dump(confusion_matrix(y_test, y_pred), "cm.pkl")
    joblib.dump(classification_report(y_test, y_pred, output_dict=True), "report.pkl")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    joblib.dump((fpr, tpr, auc(fpr,tpr)), "roc.pkl")

    if hasattr(model, "feature_importances_"):
        joblib.dump(model.feature_importances_, "feat_imp.pkl")

    st.success(f"🏆 Best Model: {model_name}")

    progress.progress(100, text="✅ Training Complete")


# =========================
# TRAIN BUTTON
# =========================
st.header("🚀 Training")

if st.button("Train Model"):
    progress = st.progress(0)
    train_all(progress)


# =========================
# LOAD MODELS
# =========================
if not os.path.exists("best_model.pkl"):
    st.warning("⚠️ Train model first")
    st.stop()

ml_model = joblib.load("best_model.pkl")
model_name = joblib.load("best_model_name.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("cols.pkl")


# =========================
# INPUT (UPLOAD ONLY)
# =========================
st.header("🎤 Upload Audio")

file = st.file_uploader("Upload WAV File", type=["wav"])

audio_bytes = None
if file:
    audio_bytes = file.read()


# =========================
# PREDICTION
# =========================
if audio_bytes:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    progress = st.progress(0)

    progress.progress(10, text="📂 Loading audio...")
    y, sr = safe_load_audio(path)

    if y is None:
        st.error("Invalid audio")
        st.stop()

    y = clean_audio(y, sr)

    progress.progress(30, text="🔍 Extracting features...")
    features = get_voice_features(path)
    df = pd.DataFrame([features]).reindex(columns=cols, fill_value=0)

    progress.progress(60, text="🤖 ML prediction...")
    ml_prob = ml_model.predict_proba(scaler.transform(df))[0][1]

    progress.progress(100, text="✅ Done")

    st.subheader("🧾 Result")
    label = "Parkinson ❌" if ml_prob > 0.5 else "Healthy ✅"

    st.success(f"Prediction: {label}")
    st.info(f"🤖 Model Used: {model_name}")

    st.metric("ML Score", f"{ml_prob*100:.2f}%")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    with colB:
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec)

        fig, ax = plt.subplots()
        librosa.display.specshow(spec, sr=sr, ax=ax)
        st.pyplot(fig)


# =========================
# CONFUSION MATRIX
# =========================
if os.path.exists("cm.pkl"):
    cm = joblib.load("cm.pkl")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)


# =========================
# ROC CURVE
# =========================
if os.path.exists("roc.pkl"):
    fpr, tpr, roc_auc = joblib.load("roc.pkl")

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.legend()
    st.pyplot(fig)


# =========================
# FEATURE IMPORTANCE
# =========================
if os.path.exists("feat_imp.pkl"):
    imp = joblib.load("feat_imp.pkl")

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    ax.bar(range(len(imp)), imp)
    st.pyplot(fig)
