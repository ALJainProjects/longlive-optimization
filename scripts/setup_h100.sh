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
echo "[1/8] Checking CUDA..."
nvidia-smi
if ! command -v nvcc &> /dev/null; then
    echo "Note: nvcc not in PATH, but CUDA drivers are available"
fi

# Create virtual environment
echo ""
echo "[2/8] Creating virtual environment..."
if [ -d ~/longlive-env ]; then
    echo "Virtual environment exists, reusing..."
else
    python3 -m venv ~/longlive-env
fi
source ~/longlive-env/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch (should match CUDA version)
echo ""
echo "[3/8] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core requirements - comprehensive list
echo ""
echo "[4/8] Installing core dependencies..."
pip install \
    pyyaml numpy scipy scikit-image tqdm einops omegaconf accelerate \
    transformers diffusers huggingface_hub sentencepiece ftfy safetensors \
    easydict timm rotary_embedding_torch requests imageio imageio-ffmpeg \
    pillow opencv-python-headless matplotlib tensorboard

# Install flash-attn (requires CUDA) - this can take a few minutes
echo ""
echo "[5/8] Installing flash-attn (this may take 5-10 minutes)..."
pip install flash-attn --no-build-isolation || {
    echo "Warning: flash-attn installation failed, trying alternative method..."
    pip install flash-attn --no-build-isolation --no-deps
}

# Install torchao for FP8 quantization (requires CUDA)
echo ""
echo "[6/8] Installing torchao for FP8 quantization..."
# First try the PyTorch CUDA index
pip install torchao --index-url https://download.pytorch.org/whl/cu121 || {
    echo "Warning: torchao from PyTorch index failed, trying PyPI..."
    pip install torchao || {
        echo "Warning: torchao installation failed completely - FP8 quantization will not be available"
    }
}
# Verify installation
python -c "import torchao; print(f'torchao version: {torchao.__version__}')" 2>/dev/null || \
    echo "Note: torchao not available - FP8 quantization will be skipped"

# Install triton for kernel fusion
echo ""
echo "[7/8] Installing triton..."
pip install triton || echo "Warning: triton installation may have issues"

# Install peft with compatible transformers version
echo ""
echo "[8/10] Installing peft with compatible transformers..."
# Pin transformers to a version compatible with peft
pip install "transformers>=4.35.0,<4.46.0" "peft>=0.6.0,<0.8.0"

# Download model weights - to correct directory
echo ""
echo "[9/10] Downloading Wan base model weights..."
pip install 'huggingface_hub[cli]'
mkdir -p wan_models longlive_models
echo "Downloading Wan2.1-T2V-1.3B from HuggingFace..."

# Use huggingface_hub Python API for more reliable downloads
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

target_dir = "./wan_models/Wan2.1-T2V-1.3B"
os.makedirs(target_dir, exist_ok=True)

print("Downloading Wan base model files...")
snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)
print("Wan base model download complete!")
EOF

# Download LongLive distilled weights (CRITICAL for performance!)
echo ""
echo "[10/10] Downloading LongLive distilled weights..."
echo "This is CRITICAL - without these, performance is 5x slower!"
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

target_dir = "./longlive_models"
os.makedirs(target_dir, exist_ok=True)

print("Downloading LongLive distilled model weights...")
snapshot_download(
    repo_id="Efficient-Large-Model/LongLive",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)
print("LongLive weights download complete!")

# Verify the critical files exist
import os
critical_files = [
    "./longlive_models/models/longlive_base.pt",
    "./longlive_models/models/lora.pt"
]
for f in critical_files:
    if os.path.exists(f):
        print(f"✓ Found: {f}")
    else:
        print(f"✗ Missing: {f}")
EOF

# Verify model download
echo ""
echo "Verifying model files..."
if [ -f "./wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" ] && \
   [ -f "./wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "✓ Model files verified successfully"
else
    echo "✗ WARNING: Some model files may be missing"
    ls -la ./wan_models/Wan2.1-T2V-1.3B/
fi

# Final verification
echo ""
echo "=============================================="
echo "Setup Verification"
echo "=============================================="
python3 << 'EOF'
import sys
print("Python version:", sys.version)

# Check key imports
modules = [
    ("torch", "PyTorch"),
    ("flash_attn", "Flash Attention"),
    ("torchao", "TorchAO (FP8)"),
    ("omegaconf", "OmegaConf"),
    ("einops", "einops"),
    ("transformers", "Transformers"),
    ("diffusers", "Diffusers"),
]

print("\nDependency check:")
for mod, name in modules:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "installed")
        print(f"  ✓ {name}: {ver}")
    except ImportError:
        print(f"  ✗ {name}: NOT INSTALLED")

# Check CUDA
import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
EOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="

# Auto-start benchmark if --run-benchmark flag is passed
if [[ "$1" == "--run-benchmark" ]] || [[ "$RUN_BENCHMARK" == "1" ]]; then
    echo ""
    echo "=============================================="
    echo "Starting Benchmark Suite..."
    echo "=============================================="
    python scripts/run_full_benchmark.py --quick
elif [[ "$1" == "--run-full-benchmark" ]] || [[ "$RUN_FULL_BENCHMARK" == "1" ]]; then
    echo ""
    echo "=============================================="
    echo "Starting Full Benchmark Suite..."
    echo "=============================================="
    python scripts/run_full_benchmark.py
else
    echo ""
    echo "To activate the environment:"
    echo "  source ~/longlive-env/bin/activate"
    echo ""
    echo "To run benchmarks:"
    echo "  python scripts/run_full_benchmark.py"
    echo ""
    echo "To run quick test:"
    echo "  python scripts/run_full_benchmark.py --quick"
    echo ""
    echo "Or re-run this script with --run-benchmark flag to auto-start"
fi
