#!/bin/bash

# Setup script for Novel View Synthesis Pipeline
# Installs all dependencies and sets up the environment

set -e  # Exit on error

echo "=========================================="
echo "Novel View Synthesis Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking for CUDA..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "Warning: CUDA/NVIDIA GPU not detected"
    echo "The pipeline will still work but training will be slow on CPU"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install numpy opencv-python pillow tqdm matplotlib scikit-image scipy pyyaml imageio imageio-ffmpeg

# Install pycolmap
echo ""
echo "Installing pycolmap..."
pip install pycolmap

# Install nerfstudio
echo ""
echo "Installing nerfstudio..."
pip install nerfstudio

# Install gsplat
echo ""
echo "Installing gsplat..."
pip install gsplat

# Install viser
echo ""
echo "Installing viser..."
pip install viser

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA version: $(python -c "import torch; print(torch.version.cuda)")"
fi

echo ""
echo "OpenCV version:"
python -c "import cv2; print(cv2.__version__)"

echo ""
echo "pycolmap installed:"
python -c "import pycolmap; print('✓')" 2>/dev/null || echo "✗"

echo ""
echo "nerfstudio installed:"
which ns-train > /dev/null && echo "✓" || echo "✗"

echo ""
echo "gsplat installed:"
python -c "import gsplat; print('✓')" 2>/dev/null || echo "✗"

echo ""
echo "viser installed:"
python -c "import viser; print('✓')" 2>/dev/null || echo "✗"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Quick start:"
echo "  1. Extract frames: python scripts/extract_frames.py video.mp4"
echo "  2. Run COLMAP: python scripts/run_colmap.py data/extracted_frames/video"
echo "  3. Process data: ns-process-data images --data data/extracted_frames/video --output-dir data/processed/video"
echo "  4. Train NeRF: python scripts/train_nerf.py data/processed/video"
echo ""
