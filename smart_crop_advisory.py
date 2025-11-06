import os
# Reduce TensorFlow logging noise (set before importing tensorflow)
# 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import kagglehub
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Step 1: Download the dataset directly from Kaggle
data_path_candidates = [
    Path("Crop_recommendation.csv"),
    Path("data") / "Crop_recommendation.csv",
]

# Try to use a local CSV if present, otherwise attempt Kaggle download
data_path = None
for p in data_path_candidates:
    if p.exists():
        data_path = str(p)
        print(f"✅ Found local dataset at: {data_path}")
        break

if data_path is None:
    try:
        print("📥 Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("varshitanalluri/crop-recommendation-dataset")
        data_path = f"{path}/Crop_recommendation.csv"
        print("✅ Dataset downloaded at:", path)
    except Exception as _e:
        raise RuntimeError("Dataset not found locally and Kaggle download failed. Please place 'Crop_recommendation.csv' in the repo root or install/configure kagglehub: " + str(_e))

# Step 2: Load the dataset
df = pd.read_csv(data_path)
# Normalize column names and strip whitespace from string columns to avoid KeyError
df.columns = df.columns.str.strip()
print("📊 Dataset loaded successfully!")
print(df.head())

# Step 3: Data preprocessing
# Detect label column robustly (dataset uses 'label' in some sources, 'Crop' in others)
label_col = None
for name in ['label', 'crop']:
    for col in df.columns:
        if col.lower() == name.lower():
            label_col = col
            break
    if label_col:
        break
if label_col is None:
    # fallback: choose a non-numeric/object column (likely the label) or the last column
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(obj_cols) > 0:
        label_col = obj_cols[-1]
    else:
        label_col = df.columns[-1]

# Strip whitespace in label values if present (e.g., 'Rice  ')
if df[label_col].dtype == object:
    df[label_col] = df[label_col].str.strip()

X = df.drop(label_col, axis=1)
y = df[label_col]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a consistent train/test split for both models using integer labels
X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# For the ANN we need categorical labels
y_train = tf.keras.utils.to_categorical(y_train_enc)
y_test = tf.keras.utils.to_categorical(y_test_enc)

# Compute class weights to help with class imbalance
classes = np.unique(y_encoded)
class_weights_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}
print("Class weights:", class_weight_dict)

# Step 4: Build the Deep Learning (ANN) Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Stacked ensemble training (K-fold stacking)
print("🚀 Preparing training pipeline (ANN +/- XGBoost)...")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Check if xgboost is available; if not, fall back to ANN-only training
try:
    from xgboost import XGBClassifier  # type: ignore
    xgb_available = True
except Exception:
    xgb_available = False
    print("⚠️ xgboost not available — proceeding with ANN-only training and skipping stacking.")

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
n_classes = len(np.unique(y_encoded))

# We'll create out-of-fold prediction matrices for the TRAINING partition (X_train)
oof_ann = np.zeros((X_train.shape[0], n_classes))
oof_xgb = np.zeros((X_train.shape[0], n_classes)) if xgb_available else None

def build_ann_model(input_dim, output_dim):
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

if xgb_available:
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
        print(f"-- Fold {fold+1}/{K}")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr_enc_fold, y_val_enc_fold = y_train_enc[tr_idx], y_train_enc[val_idx]
        y_tr_cat = tf.keras.utils.to_categorical(y_tr_enc_fold)

        # ANN for this fold
        ann = build_ann_model(X_tr.shape[1], n_classes)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        ann.fit(X_tr, y_tr_cat, validation_data=(X_val, tf.keras.utils.to_categorical(y_val_enc_fold)),
                epochs=60, batch_size=64, callbacks=callbacks, class_weight=class_weight_dict, verbose=0)
        oof_ann[val_idx] = ann.predict(X_val)

        # XGBoost for this fold
        xgb_fold = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        sample_weight_fold = np.array([class_weight_dict[int(c)] for c in y_tr_enc_fold])
        xgb_fold.fit(X_tr, y_tr_enc_fold, sample_weight=sample_weight_fold, verbose=False)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_val)
else:
    print("Skipping K-fold stacking because xgboost is not available.")

meta_clf = None
if xgb_available:
    # Train meta-learner on OOF predictions
    meta_X = np.hstack([oof_ann, oof_xgb])
    meta_y = y_train_enc
    meta_clf = LogisticRegression(max_iter=2000)
    meta_clf.fit(meta_X, meta_y)
else:
    print("Meta-learner not trained (requires xgboost stacking). Predictions will use ANN only.")

# Train final base models on the full training partition
print("\nTraining final base models on full training set...")
final_ann = build_ann_model(X_train.shape[1], n_classes)
final_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
# Use fewer epochs by default to keep runs quicker; user can increase for better accuracy
history = final_ann.fit(
    X_train,
    tf.keras.utils.to_categorical(y_train_enc),
    epochs=20,
    batch_size=64,
    callbacks=final_callbacks,
    class_weight=class_weight_dict,
    validation_data=(X_test, tf.keras.utils.to_categorical(y_test_enc)),
    verbose=1,
)

# Persist training curves for quick inspection of model performance over epochs.
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get('accuracy', []), label='train')
    axes[0].plot(history.history.get('val_accuracy', []), label='validation')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history.get('loss', []), label='train')
    axes[1].plot(history.history.get('val_loss', []), label='validation')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    curves_path = Path('models') / 'training_curves.png'
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"📈 Saved training curves to: {curves_path}")

    # Also persist the raw history so it can be inspected or plotted elsewhere.
    history_path = Path('models') / 'training_history.csv'
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"🗒️ Saved detailed training history to: {history_path}")
except Exception as _plot_err:
    print(f"⚠️ Could not generate training curve plot: {_plot_err}")

final_xgb = None
if xgb_available:
    final_xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    sample_weight_full = np.array([class_weight_dict[int(c)] for c in y_train_enc])
    final_xgb.fit(X_train, y_train_enc, sample_weight=sample_weight_full, verbose=False)

# Evaluate ensemble on held-out test set
# Evaluate ensemble or ANN-only on held-out test set
ann_test_proba = final_ann.predict(X_test)
if xgb_available and final_xgb is not None and meta_clf is not None:
    xgb_test_proba = final_xgb.predict_proba(X_test)
    meta_test_X = np.hstack([ann_test_proba, xgb_test_proba])
    meta_test_preds = meta_clf.predict(meta_test_X)
    ens_test_acc = accuracy_score(y_test_enc, meta_test_preds)
    print(f"✅ Stacked Ensemble Test Accuracy: {ens_test_acc * 100:.2f}%")
else:
    ann_preds = np.argmax(ann_test_proba, axis=1)
    ann_acc = accuracy_score(y_test_enc, ann_preds)
    print(f"✅ ANN-only Test Accuracy: {ann_acc * 100:.2f}%")

# Save final artifacts
# Save artifacts (use native Keras format for the ANN to avoid legacy HDF5 warnings)
os.makedirs("models", exist_ok=True)
ann_keras_path = "models/crop_model.keras"
final_ann.save(ann_keras_path)
if final_xgb is not None:
    try:
        joblib.dump(final_xgb, "models/xgb_model.joblib")
    except Exception:
        print("⚠️ Could not save XGBoost model artifact; continuing.")
try:
    joblib.dump(encoder, "models/label_encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
except Exception:
    print("⚠️ Could not save encoder/scaler artifacts; continuing.")
if meta_clf is not None:
    try:
        joblib.dump(meta_clf, "models/meta_model.joblib")
    except Exception:
        print("⚠️ Could not save meta_model artifact; continuing.")
# Save feature names and a small background sample for SHAP
try:
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.joblib")
    # background: up to 100 examples from the scaled training set used by models
    bg_size = min(100, X_train.shape[0])
    rng = np.random.default_rng(42)
    idxs = rng.choice(X_train.shape[0], size=bg_size, replace=False)
    shap_bg = X_train[idxs]
    joblib.dump(shap_bg, "models/shap_background.joblib")
except Exception:
    # non-critical: continue even if saving these fails
    pass

print("💾 Saved stacked ensemble artifacts to models/ (crop_model.keras, xgb_model.joblib, label_encoder.joblib, scaler.joblib, meta_model.joblib)")
