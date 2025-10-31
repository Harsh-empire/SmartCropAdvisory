#!/usr/bin/env bash
# Simple helper for Unix-like shells (bash, WSL, Git Bash)
# Usage:
#   ./run.sh           # create venv if missing, install deps, run app
#   ./run.sh train     # run training (smart_crop_advisory.py) instead of the app

MODE=${1:-run}
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating venv..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements (if present)..."
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  echo "No requirements.txt found — skipping pip install."
fi

if [ "$MODE" = "train" ]; then
  echo "Running training script..."
  python smart_crop_advisory.py
else
  echo "Starting Streamlit app..."
  python -m streamlit run app.py
fi
