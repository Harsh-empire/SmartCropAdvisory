import os
# Reduce TensorFlow logging noise before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
try:
    # absl logging uses its own verbosity API; set it to ERROR to suppress the compiled-model warning
    import absl.logging as _absl_logging

    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass

try:
    import streamlit as st
except Exception as e:
    raise ImportError(
        "streamlit is required to run the app. Install it in your environment, e.g. 'pip install streamlit'. Original error: "
        + str(e)
    )
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib
import ast
from pathlib import Path
import shutil
import datetime


def normalize_bg(candidate):
    """Normalize various saved background formats into a numeric 2D numpy array or return None on failure.

    Handles:
    - pandas DataFrame
    - numpy arrays of numeric types
    - numpy arrays / lists of strings that represent lists (e.g. "[1,2,3]")
    - lists of lists
    """
    try:
        import pandas as _pd

        if isinstance(candidate, _pd.DataFrame):
            return _pd.DataFrame(candidate).to_numpy(dtype=float)
    except Exception:
        pass

    try:
        arr = np.asarray(candidate)
        if arr.dtype.type is np.str_ or arr.dtype == object:
            parsed = []
            for el in arr:
                if isinstance(el, str):
                    try:
                        val = ast.literal_eval(el)
                    except Exception:
                        parts = str(el).strip().lstrip('[').rstrip(']').split(',')
                        val = [float(p) for p in parts if p != '']
                    parsed.append(val)
                else:
                    parsed.append(el)
            return np.asarray(parsed, dtype=float)
        else:
            return arr.astype(float)
    except Exception:
        return None


@st.cache_resource
def load_model():
    # Prefer loading pre-saved preprocessing and model artifacts from ./models
    models_dir = Path("models")
    # prefer native Keras format, fall back to legacy HDF5
    model_keras_path = models_dir / "crop_model.keras"
    model_path = models_dir / "crop_model.h5"
    xgb_path = models_dir / "xgb_model.joblib"
    encoder_path = models_dir / "label_encoder.joblib"
    scaler_path = models_dir / "scaler.joblib"
    meta_path = models_dir / "meta_model.joblib"

    # model (.keras preferred, .h5 fallback) is required for inference
    if not model_keras_path.exists() and not model_path.exists():
        st.error(
            f"Model file not found in models/.\nRun `python smart_crop_advisory.py` from the repo root to train and save the model and artifacts (this creates crop_model.keras and other files)."
        )
        st.stop()

    chosen_path = str(model_keras_path) if model_keras_path.exists() else str(model_path)

    # load Keras model without recompiling to avoid compiled-metrics warning when only inferencing
    try:
        model = tf.keras.models.load_model(chosen_path, compile=False)
    except TypeError:
        # older TF versions may not accept compile=False; fall back to default load
        model = tf.keras.models.load_model(chosen_path)

    # Load encoder and scaler if available; otherwise attempt to reconstruct them from the dataset
    le = None
    scaler = None
    missing_preproc = []
    if encoder_path.exists() and scaler_path.exists():
        try:
            le = joblib.load(str(encoder_path))
            scaler = joblib.load(str(scaler_path))
        except Exception:
            missing_preproc = [encoder_path, scaler_path]
    else:
        missing_preproc = [p for p in (encoder_path, scaler_path) if not p.exists()]

    if le is None or scaler is None:
        # fallback: rebuild encoder & scaler from the Kaggle dataset (network download)
        st.warning("Label encoder or scaler artifact not found. Rebuilding from the dataset (this will download the dataset)...")
        path = kagglehub.dataset_download("varshitanalluri/crop-recommendation-dataset")
        df = pd.read_csv(f"{path}/Crop_recommendation.csv")
        df.columns = df.columns.str.strip()
        # detect label column
        label_col = None
        for name in ['label', 'crop']:
            for col in df.columns:
                if col.lower() == name.lower():
                    label_col = col
                    break
            if label_col:
                break
        if label_col is None:
            obj_cols = df.select_dtypes(include=['object']).columns.tolist()
            label_col = obj_cols[-1] if len(obj_cols) > 0 else df.columns[-1]
        y = df[label_col]
        if y.dtype == object:
            y = y.str.strip()
        X = df.drop(label_col, axis=1)
        le = LabelEncoder().fit(y)
        scaler = StandardScaler().fit(X)
        # optionally persist them for next runs
        try:
            joblib.dump(le, str(encoder_path))
            joblib.dump(scaler, str(scaler_path))
        except Exception:
            # if we can't write, continue with in-memory objects
            pass

    # Load XGBoost model if present; if not, we'll use ANN only
    xgb = None
    if xgb_path.exists():
        try:
            xgb = joblib.load(str(xgb_path))
        except Exception:
            xgb = None

    # Load stacking meta-learner if present
    meta = None
    if meta_path.exists():
        try:
            meta = joblib.load(str(meta_path))
        except Exception:
            meta = None

    # Load feature names and shap background if available
    feature_names = None
    shap_bg = None
    feature_path = models_dir / "feature_names.joblib"
    shap_bg_path = models_dir / "shap_background.joblib"
    if feature_path.exists():
        try:
            feature_names = joblib.load(str(feature_path))
        except Exception:
            feature_names = None
    if shap_bg_path.exists():
        try:
            raw_bg = joblib.load(str(shap_bg_path))
            shap_bg = normalize_bg(raw_bg)
        except Exception:
            shap_bg = None

    return model, xgb, le, scaler, meta, feature_names, shap_bg

model, xgb_model, label_enc, scaler, meta_model, feature_names, shap_bg = load_model()

st.title("Smart Crop Advisory")
st.write("Enter the soil & weather parameters below:")

N  = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0)
P  = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=42.0)
K  = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=43.0)
temp     = st.number_input("Temperature (°C)",    min_value=0.0, max_value=50.0, value=20.88)
humidity = st.number_input("Humidity (%)",         min_value=0.0, max_value=100.0, value=82.0)
ph       = st.number_input("Soil pH",               min_value=0.0, max_value=14.0,  value=6.5)
rainfall = st.number_input("Rainfall (mm)",        min_value=0.0, max_value=500.0, value=202.93)

input_array = np.array([[N, P, K, temp, humidity, ph, rainfall]])
# Avoid sklearn warning about feature names by passing a DataFrame with the same columns when possible
try:
    if feature_names is not None and len(feature_names) == input_array.shape[1]:
        input_df = pd.DataFrame(input_array, columns=feature_names)
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = scaler.transform(input_array)
except Exception:
    # fallback
    input_scaled = scaler.transform(input_array)

# ANN probabilities
ann_proba = model.predict(input_scaled)
if meta_model is not None and xgb_model is not None:
    # Use meta-learner on stacked probabilities (ANN probs concat XGB probs)
    try:
        xgb_proba = xgb_model.predict_proba(input_scaled)
        stacked_input = np.hstack([ann_proba, xgb_proba])
        meta_pred = meta_model.predict(stacked_input)
        crop_idx = int(meta_pred[0])
        recommended_crop = label_enc.inverse_transform([crop_idx])[0]
    except Exception:
        # fallback to averaging if anything fails
        try:
            xgb_proba = xgb_model.predict_proba(input_scaled)
            avg_proba = (ann_proba + xgb_proba) / 2.0
            crop_idx = np.argmax(avg_proba, axis=1)[0]
            recommended_crop = label_enc.inverse_transform([crop_idx])[0]
        except Exception:
            recommended_crop = label_enc.inverse_transform([np.argmax(ann_proba, axis=1)[0]])[0]
elif xgb_model is not None:
    try:
        xgb_proba = xgb_model.predict_proba(input_scaled)
        avg_proba = (ann_proba + xgb_proba) / 2.0
        crop_idx = np.argmax(avg_proba, axis=1)[0]
        recommended_crop = label_enc.inverse_transform([crop_idx])[0]
    except Exception:
        recommended_crop = label_enc.inverse_transform([np.argmax(ann_proba, axis=1)[0]])[0]
else:
    crop_idx = np.argmax(ann_proba, axis=1)[0]
    recommended_crop = label_enc.inverse_transform([crop_idx])[0]

st.success(f"🌾 Recommended Crop: **{recommended_crop}**")
