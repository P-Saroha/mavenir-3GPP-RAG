#!/usr/bin/env bash
# setup.sh — Linux / macOS bootstrap
# Usage:  bash setup.sh
# Creates a Python 3.11 virtual environment, installs all dependencies,
# and copies .env.example → .env if no .env exists yet.

set -euo pipefail

echo "==> Checking Python version..."
python3 --version

# Create venv if it doesn't already exist
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "==> Virtual environment already exists, skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip
echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

# Install CPU-only PyTorch first (avoids pulling the large CUDA build)
echo "==> Installing CPU-only PyTorch..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --quiet

# Install remaining dependencies
echo "==> Installing project dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# Install project in editable mode so local packages are importable
echo "==> Installing project in editable mode..."
pip install -e . --quiet

# Copy .env.example → .env if needed
if [ ! -f ".env" ]; then
    echo "==> Creating .env from .env.example — fill in your API keys!"
    cp .env.example .env
else
    echo "==> .env already exists, skipping copy."
fi

echo ""
echo "==> Setup complete!"
echo "    Activate the venv with:  source .venv/bin/activate"
echo "    Then edit .env with your GROK_API_KEY and run:"
echo "      pytest tests/  -- to verify the installation"
