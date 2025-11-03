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

with st.expander("Why this prediction? (Explainable ML using SHAP)"):
    if xgb_model is None:
        st.info("XGBoost model not available — SHAP explanations require the tree model. Train the full pipeline to enable explanations.")
    else:
        try:
            import shap  # type: ignore[reportMissingImports]
            import matplotlib.pyplot as plt
            # --- Diagnostics: show loaded SHAP background and feature names for debugging ---
            with st.expander("SHAP background diagnostics"):
                try:
                    if shap_bg is None:
                        st.write("shap_background: None (no background saved)")
                    else:
                        arr = np.asarray(shap_bg)
                        st.write("type:", type(shap_bg))
                        st.write("numpy shape:", arr.shape)
                        st.write("dtype:", arr.dtype)
                        # show a concise preview of the first row
                        if arr.size > 0:
                            try:
                                preview = arr[0].tolist() if arr.ndim >= 1 else arr.tolist()
                            except Exception:
                                preview = str(arr[0])
                            st.write("first row preview:", preview)

                    # Repair button: attempt to reload, normalize, and re-save the background
                    if shap_bg is not None:
                        if st.button("Repair shap_background (normalize & overwrite)"):
                                try:
                                    raw_path = Path('models') / 'shap_background.joblib'
                                    backup_dir = Path('models') / 'backups'
                                    try:
                                        backup_dir.mkdir(parents=True, exist_ok=True)
                                    except Exception:
                                        # proceed even if backup dir can't be created
                                        pass

                                    # create a timestamped backup of the original file before modifying
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                    backup_path = backup_dir / f"shap_background.{timestamp}.joblib"
                                    try:
                                        if raw_path.exists():
                                            shutil.copy2(str(raw_path), str(backup_path))
                                            st.info(f"Backup of original shap_background created at: {backup_path}")
                                    except Exception as _copy_err:
                                        st.warning(f"Could not create backup of original shap_background: {_copy_err}")

                                    raw = joblib.load(str(raw_path))
                                    repaired = normalize_bg(raw)
                                    if repaired is None:
                                        st.error("Repair failed: could not coerce background to numeric array")
                                    else:
                                        joblib.dump(repaired, str(raw_path))
                                        st.success(f"shap_background.joblib normalized and re-saved. Backup (if created): {backup_path}")
                                        # Try to programmatically rerun the app; Streamlit API differs across versions
                                        try:
                                            if hasattr(st, 'experimental_rerun'):
                                                st.experimental_rerun()
                                            elif hasattr(st, 'rerun'):
                                                st.rerun()
                                            else:
                                                st.info('Please reload the app to apply the repaired background (browser refresh).')
                                        except Exception:
                                            st.info('Please reload the app to apply the repaired background (browser refresh).')
                                except Exception as _e:
                                    st.error("Repair attempt failed: " + str(_e))
                    else:
                        if st.button("Create normalized shap_background from current models/scaler"):
                            # Attempt to create a simple numeric background in the scaled feature space.
                            try:
                                if xgb_model is None:
                                    st.error("XGBoost model is required to create a SHAP background. Train the full pipeline first.")
                                else:
                                    # input_scaled is available above; use its shape to determine number of features
                                    try:
                                        n_feat = input_scaled.shape[1]
                                    except Exception:
                                        # fallback: try feature_names length
                                        n_feat = len(feature_names) if feature_names is not None else 7

                                    # Create a simple zero-centered background in the scaled space (mean-like)
                                    bg_size = 50
                                    bg = np.zeros((bg_size, n_feat), dtype=float)
                                    models_dir = Path('models')
                                    models_dir.mkdir(parents=True, exist_ok=True)
                                    bg_path = models_dir / 'shap_background.joblib'
                                    joblib.dump(bg, str(bg_path))
                                    st.success(f"Created numeric SHAP background at: {bg_path}")
                                    # Try to programmatically rerun the app so SHAP picks up the new file
                                    try:
                                        if hasattr(st, 'experimental_rerun'):
                                            st.experimental_rerun()
                                        elif hasattr(st, 'rerun'):
                                            st.rerun()
                                        else:
                                            st.info('Please refresh the browser to reload the app and apply the new SHAP background.')
                                    except Exception:
                                        st.info('Please refresh the browser to reload the app and apply the new SHAP background.')
                            except Exception as _e:
                                st.error("Could not create SHAP background: " + str(_e))
                except Exception as _diag_e:
                    st.write("Could not inspect shap_background:", str(_diag_e))

            with st.expander("SHAP feature names diagnostics"):
                try:
                    if feature_names is None:
                        st.write("feature_names: None")
                    else:
                        st.write("type:", type(feature_names))
                        try:
                            st.write("count:", len(feature_names))
                            st.write("first features:", feature_names[:10])
                        except Exception:
                            st.write(feature_names)
                except Exception as _diag_e:
                    st.write("Could not inspect feature_names:", str(_diag_e))

            # Ensure numeric arrays (sometimes joblib loads may give object dtype)
            try:
                safe_input = np.asarray(input_scaled, dtype=float)
            except Exception as _e:
                raise RuntimeError(f"SHAP input conversion failed: {str(_e)}")

            if shap_bg is not None:
                try:
                    safe_bg = np.asarray(shap_bg, dtype=float)
                except Exception:
                    safe_bg = None
            else:
                safe_bg = None

            # Build a TreeExplainer for the XGBoost model. Use the saved background if available for stable attribution.
            if safe_bg is not None:
                explainer = shap.TreeExplainer(xgb_model, data=safe_bg)
            else:
                explainer = shap.TreeExplainer(xgb_model)

            shap_values = explainer.shap_values(safe_input)

            # Normalize shap_values into a (n_classes, n_samples, n_features)-like structure
            if isinstance(shap_values, list):
                # list per class: each element shape (n_samples, n_features)
                try:
                    class_shap = np.asarray(shap_values[int(crop_idx)])[0]
                except Exception:
                    # fallback: take first class's explanation
                    class_shap = np.asarray(shap_values[0])[0]
            else:
                # Could be ndarray shape (n_samples, n_features) for single-output
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    # shape: (n_classes, n_samples, n_features)
                    class_shap = arr[int(crop_idx), 0, :]
                elif arr.ndim == 2:
                    # shape: (n_samples, n_features)
                    class_shap = arr[0]
                else:
                    raise RuntimeError("Unexpected SHAP values shape: " + str(arr.shape))

            fnames = feature_names if feature_names is not None else [f"f{i}" for i in range(class_shap.shape[0])]

            # Bar chart of SHAP contributions for this prediction
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh(fnames, class_shap)
            ax.set_title(f"SHAP feature contributions for '{recommended_crop}'")
            ax.set_xlabel("SHAP value (impact on model output)")
            st.pyplot(fig)

            # Show a sorted table of contributions
            contrib_df = pd.DataFrame({"feature": fnames, "shap_value": class_shap})
            st.table(contrib_df.sort_values("shap_value", ascending=False).reset_index(drop=True))
        except Exception as e:
            # Show a short, friendly message to users and hide the raw exception by default.
            st.error("SHAP explanations are unavailable. Ensure 'shap' and 'xgboost' are installed or repair the SHAP background via diagnostics.")
            with st.expander("Show debug details"):
                import traceback

                st.write(str(e))
                st.write(traceback.format_exc())
