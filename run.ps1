<#
Simple helper for Windows PowerShell to create a venv, install requirements, and run the Streamlit app.
Usage:
  ./run.ps1            # create venv if missing, install deps, run app
  ./run.ps1 train      # run training (smart_crop_advisory.py) instead of the app

Note: This script sets ExecutionPolicy for the current process before activating the venv.
#>
param(
    [string]$Mode = "run"  # 'run' or 'train'
)

Write-Host "Checking .venv..."
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

Write-Host "Allowing script execution for this session..."
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

Write-Host "Activating virtual environment..."
& '.\.venv\Scripts\Activate.ps1'

Write-Host "Upgrading pip and installing requirements (if present)..."
python -m pip install --upgrade pip
if (Test-Path -Path .\requirements.txt) {
    python -m pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found — skipping pip install."
}

if ($Mode -eq 'train') {
    Write-Host "Running training script..."
    python smart_crop_advisory.py
} else {
    Write-Host "Starting Streamlit app..."
    python -m streamlit run app.py
}
<#
Simple helper for Windows PowerShell to create a venv, install requirements, and run the Streamlit app.
Usage:
  ./run.ps1            # create venv if missing, install deps, run app
  ./run.ps1 train      # run training (smart_crop_advisory.py) instead of the app

Note: This script sets ExecutionPolicy for the current process before activating the venv.
#>
param(
    [string]$Mode = "run"  # 'run' or 'train'
)

Write-Host "Checking .venv..."
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

Write-Host "Allowing script execution for this session..."
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

Write-Host "Activating virtual environment..."
& '.\.venv\Scripts\Activate.ps1'

Write-Host "Upgrading pip and installing requirements (if present)..."
python -m pip install --upgrade pip
if (Test-Path -Path .\requirements.txt) {
    python -m pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found — skipping pip install."
}

if ($Mode -eq 'train') {
    Write-Host "Running training script..."
    python smart_crop_advisory.py
} else {
    Write-Host "Starting Streamlit app..."
    python -m streamlit run app.py
}
