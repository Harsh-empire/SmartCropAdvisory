# Smart Crop Advisory

A small ML + Streamlit project that trains a crop recommendation model from the Kaggle "Crop Recommendation" dataset and serves predictions via a Streamlit app.

This repo contains:
- `smart_crop_advisory.py` — training script (builds ANN + XGBoost stacking ensemble and saves artifacts to `models/`).
- `app.py` — Streamlit app that loads saved artifacts (ANN, XGBoost, preprocessing) and serves predictions.
- `create_artifacts.py` — helper to build and persist preprocessing artifacts (encoder + scaler) without training full models.

Quick start (Windows PowerShell)
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv venv
# If your PowerShell blocks script execution, run once (in this terminal only):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

3. (Optional) Create preprocessing artifacts only — useful if you want the app to run quickly without training:

```powershell
python create_artifacts.py
```

4. Train the stacked ensemble (this will download the dataset via Kaggle, train models, and save artifacts to `models/`):

```powershell
python smart_crop_advisory.py
```

5. Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

Troubleshooting
- If VS Code reports `Import "..." could not be resolved` (Pylance): make sure VS Code is using the repo venv Python. Open Command Palette → **Python: Select Interpreter** → choose the interpreter at `./venv/Scripts/python.exe`.
- If you see PowerShell execution policy errors when activating the venv, run the `Set-ExecutionPolicy` line shown above in the same terminal once.
- If the app says a model or preprocessing artifact is missing:
  - Run `python create_artifacts.py` to create encoder/scaler.
  - Or run `python smart_crop_advisory.py` to train and create all artifacts.
- If you want explainability (SHAP) visualizations, make sure `shap` and `xgboost` are installed (they're in `requirements.txt`). If SHAP isn't installed, the app will still run but explanations won't be shown.

Files written to `models/` by training
- `crop_model.h5` — Keras ANN model
- `xgb_model.joblib` — trained XGBoost model
- `label_encoder.joblib` — LabelEncoder for mapping indices→crop names
- `scaler.joblib` — StandardScaler for input features
- `meta_model.joblib` — LogisticRegression meta-learner for stacking
- `feature_names.joblib`, `shap_background.joblib` — optional artifacts used by SHAP explainability

Development notes
- The model input contract is unchanged: features must be in order `[N, P, K, temperature, humidity, ph, rainfall]`.
- For reproducible inference, keep the `models/` artifacts committed or stored in CI/artifact storage. The app will attempt to rebuild encoder/scaler by re-downloading the dataset if missing, but it's best to persist artifacts.

If you'd like, I can add a `Dockerfile` and a CI job that trains the models and uploads artifacts to release assets or cloud storage.

---
Happy to walk through any part of this setup or add Docker/CI next.
