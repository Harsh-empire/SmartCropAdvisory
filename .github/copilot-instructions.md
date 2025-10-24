## Smart Crop Advisory — Copilot instructions

Short, specific guidance for an AI coding assistant working on this repository.

- Project overview
  - This is a small ML + Streamlit project. Training logic lives in `smart_crop_advisory.py`; the web UI lives in `app.py`.
  - `smart_crop_advisory.py` downloads the Kaggle dataset (via `kagglehub.dataset_download`), trains a Keras model and saves it to `models/crop_model.h5`.
  - `app.py` is the Streamlit app. It reconstructs a `LabelEncoder` and `StandardScaler` by re-downloading the dataset, then loads the saved Keras model for inference.

- Key files and patterns (explicit examples)
  - `app.py`:
    - `@st.cache(allow_output_mutation=True) def load_model():` — the app caches model+encoder+scaler but re-creates encoder/scaler from the dataset every startup.
    - Loads model from `models/crop_model.h5` (relative path).
    - Prediction path: scale input -> `model.predict` -> `np.argmax(pred)` -> `label_enc.inverse_transform([idx])`.
  - `smart_crop_advisory.py`:
    - Uses `kagglehub.dataset_download("varshitanalluri/crop-recommendation-dataset")` to obtain `Crop_recommendation.csv`.
    - Fits `LabelEncoder` and `StandardScaler`, trains a Keras Sequential model and calls `model.save("models/crop_model.h5")`.

- Why this structure matters
  - The model file contains only the Keras weights/architecture. The LabelEncoder and Scaler are not serialized — they are rebuilt from the dataset at runtime. Any change to the dataset or label ordering will change inference mapping. Keep dataset source stable.
  - Training and serving are intentionally separated: training script is runnable from the repo root (`python smart_crop_advisory.py`), while serving is a Streamlit app (`streamlit run app.py`).

- Developer workflows (commands)
  - Train model and create `models/crop_model.h5` (PowerShell):
    ```powershell
    python smart_crop_advisory.py
    ```
  - Run the Streamlit app locally:
    ```powershell
    streamlit run app.py
    ```
  - If Kaggle downloads fail or you want offline runs: place `Crop_recommendation.csv` at the path the code expects (the train script expects it at `<downloaded_path>/Crop_recommendation.csv`), or modify `app.py`/`smart_crop_advisory.py` to point to a local CSV.

- Dependencies (discoverable from imports)
  - streamlit, pandas, numpy, kagglehub, scikit-learn, tensorflow (keras), matplotlib
  - There is no requirements.txt in the repo; create one or install these packages in the environment before running.

- Project-specific pitfalls and tips
  - Do NOT assume LabelEncoder/scaler are persisted: changes to dataset source/order will change label indices. If you need reproducible inference, serialize encoder/scaler (e.g., with joblib/pickle) and load them in `app.py` instead of re-fitting.
  - `app.py` uses `st.cache(allow_output_mutation=True)` to cache the loaded model. TensorFlow objects can be sensitive to caching — if you see strange session/thread errors, restart the Streamlit process.
  - Model file path is relative; CI or other automation must run from the repo root or adjust the path.
  - `kagglehub.dataset_download` is a network dependency. Tests/CI should mock or cache the dataset download step.

- When making edits
  - Preserve the inference contract: inputs are 7 floats in order [N, P, K, temperature, humidity, ph, rainfall]. The network outputs softmax over crops; code uses argmax then `LabelEncoder.inverse_transform`.
  - If you change training label encoding or class order, update how app reconstructs/loads the encoder to keep parity.

If anything above is unclear or you'd like me to expand examples (e.g., add a serialized-encoder example or a small requirements.txt), tell me which part to iterate on.
