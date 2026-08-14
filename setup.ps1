# setup.ps1 - Windows PowerShell bootstrap
# Usage:  .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python version..." -ForegroundColor Cyan
$pyver = python --version 2>&1
Write-Host "    Found: $pyver"

if (-Not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "==> Virtual environment already exists, skipping." -ForegroundColor Yellow
}

Write-Host "==> Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host "==> Installing CPU-only PyTorch..." -ForegroundColor Cyan
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --quiet

Write-Host "==> Installing project dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

Write-Host "==> Installing project in editable mode..." -ForegroundColor Cyan
pip install -e . --quiet

if (-Not (Test-Path ".env")) {
    Write-Host "==> Creating .env from .env.example" -ForegroundColor Green
    Copy-Item .env.example .env
} else {
    Write-Host "==> .env already exists, skipping." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==> Setup complete!" -ForegroundColor Green
Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Test:     pytest tests/ -v"
