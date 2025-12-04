#!/bin/bash
# LongLive H100 Setup Script
# Run this after provisioning a Lambda Labs H100 instance
# Usage: bash scripts/setup_h100.sh

set -e

echo "=============================================="
echo "LongLive H100 Setup Script"
echo "=============================================="

# Check CUDA
echo ""
echo "[1/6] Checking CUDA..."
nvidia-smi
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please ensure CUDA is installed."
    exit 1
fi
nvcc --version

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
python3 -m venv ~/longlive-env
source ~/longlive-env/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch (should match CUDA version)
echo ""
echo "[3/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core requirements
echo ""
echo "[4/6] Installing requirements..."
pip install -r requirements.txt

# Install flash-attn (requires CUDA)
echo ""
echo "[5/6] Installing flash-attn..."
pip install flash-attn --no-build-isolation

# Install torchao for quantization
echo ""
echo "[5/6] Installing torchao..."
pip install torchao

# Install development dependencies
pip install -e ".[dev,benchmark]"

# Download model weights
echo ""
echo "[6/6] Downloading model weights..."
pip install huggingface_hub[cli]
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --local-dir ./longlive_models

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source ~/longlive-env/bin/activate"
echo ""
echo "To run benchmarks:"
echo "  python scripts/run_full_benchmark.py"
echo ""
echo "To run quick test:"
echo "  python benchmarks/run_benchmark.py --config quick"
echo ""
