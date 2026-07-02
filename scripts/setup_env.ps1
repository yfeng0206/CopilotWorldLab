# CopilotWorldLab - local environment setup (Windows, PowerShell).
#
# Reproduces the verified setup on Windows 11 + RTX 3090 (24 GB):
#   - Python 3.11 virtual environment in .venv
#   - PyTorch + torchvision from the CUDA 12.4 wheel index
#   - simulation / world-model / test dependencies from requirements.txt
#
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# This script does NOT download model checkpoints; run scripts\download_checkpoints.py
# for that.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
}
Write-Host "Using Python: $python"
& $python --version

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv"
    & $python -m venv .venv
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch (CUDA 12.4 wheels)"
& $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing project dependencies"
& $venvPy -m pip install -r requirements.txt

Write-Host "Verifying Torch + CUDA"
& $venvPy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host "Done. Activate with: .venv\Scripts\Activate.ps1"
