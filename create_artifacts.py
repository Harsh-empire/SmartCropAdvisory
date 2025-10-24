import os
import joblib
import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder

"""
Utility script: download dataset, build LabelEncoder and StandardScaler, and save them
into the repository `models/` folder so the Streamlit app doesn't need to re-download
or re-fit at startup.

Run:
    python create_artifacts.py

This will create:
 - models/label_encoder.joblib
 - models/scaler.joblib

If you already have `models/crop_model.h5` and `models/xgb_model.joblib` saved,
this script will not touch them.
"""

def main():
    print("📥 Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("varshitanalluri/crop-recommendation-dataset")
    print("✅ Dataset downloaded at:", path)

    data_path = f"{path}/Crop_recommendation.csv"
    df = pd.read_csv(data_path)
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

    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.strip()

    X = df.drop(label_col, axis=1)
    y = df[label_col]

    le = LabelEncoder().fit(y)
    scaler = StandardScaler().fit(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    print("💾 Saved label_encoder.joblib and scaler.joblib to models/")

if __name__ == '__main__':
    main()
