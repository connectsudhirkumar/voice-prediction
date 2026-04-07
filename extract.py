import parselmouth
import numpy as np

def get_voice_features(file_path):

    snd = parselmouth.Sound(file_path)

    # Pitch + Pulses
    pitch = snd.to_pitch()
    pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

    features = {}

    # =========================
    # FREQUENCY FEATURES
    # =========================
    features["MDVP:Fo(Hz)"] = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    features["MDVP:Fhi(Hz)"] = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    features["MDVP:Flo(Hz)"] = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # =========================
    # JITTER FEATURES
    # =========================
    features["MDVP:Jitter(%)"] = parselmouth.praat.call(
        pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    features["MDVP:Jitter(Abs)"] = parselmouth.praat.call(
        pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
    )

    features["MDVP:RAP"] = parselmouth.praat.call(
        pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )

    features["MDVP:PPQ"] = parselmouth.praat.call(
        pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )

    features["Jitter:DDP"] = 3 * features["MDVP:RAP"]

    # =========================
    # SHIMMER FEATURES
    # =========================
    features["MDVP:Shimmer"] = parselmouth.praat.call(
        [snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["MDVP:Shimmer(dB)"] = parselmouth.praat.call(
        [snd, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["Shimmer:APQ3"] = parselmouth.praat.call(
        [snd, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["Shimmer:APQ5"] = parselmouth.praat.call(
        [snd, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["MDVP:APQ"] = parselmouth.praat.call(
        [snd, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["Shimmer:DDA"] = 3 * features["Shimmer:APQ3"]

    # =========================
    # NOISE FEATURES
    # =========================
    harmonicity = snd.to_harmonicity()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    features["HNR"] = hnr if hnr is not None else 0
    features["NHR"] = 1 / (features["HNR"] + 1e-6)

    # =========================
    # SAFE FALLBACK (important)
    # =========================
    for key in features:
        if features[key] is None or np.isnan(features[key]):
            features[key] = 0

    return features