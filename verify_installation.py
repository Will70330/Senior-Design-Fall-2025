#!/usr/bin/env python3
"""
Verification script to check all dependencies are installed correctly
"""

import sys
from pathlib import Path

def check_package(name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = name

    try:
        __import__(import_name)
        print(f"✓ {name}")
        return True
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        return False

def check_command(cmd):
    """Check if a command is available"""
    import shutil
    if shutil.which(cmd):
        print(f"✓ {cmd}")
        return True
    else:
        print(f"✗ {cmd} - NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("Novel View Synthesis Pipeline - Installation Verification")
    print("=" * 60)
    print()

    all_good = True

    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    print()

    # Check core packages
    print("Core Python Packages:")
    all_good &= check_package("numpy")
    all_good &= check_package("opencv-python", "cv2")
    all_good &= check_package("scipy")
    all_good &= check_package("matplotlib")
    all_good &= check_package("scikit-image", "skimage")
    all_good &= check_package("pyyaml", "yaml")
    all_good &= check_package("imageio")
    all_good &= check_package("tqdm")
    print()

    # Check PyTorch
    print("Deep Learning Frameworks:")
    torch_ok = check_package("torch")
    if torch_ok:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    all_good &= torch_ok
    all_good &= check_package("torchvision")
    print()

    # Check novel view synthesis packages
    print("Novel View Synthesis Libraries:")
    all_good &= check_package("pycolmap")
    all_good &= check_package("gsplat")
    all_good &= check_package("viser")
    all_good &= check_package("nerfstudio")
    print()

    # Check NeRFStudio commands
    print("NeRFStudio Commands:")
    all_good &= check_command("ns-train")
    all_good &= check_command("ns-process-data")
    all_good &= check_command("ns-export")
    all_good &= check_command("ns-render")
    print()

    # Check COLMAP
    print("External Dependencies:")
    colmap_ok = check_command("colmap")
    if not colmap_ok:
        print("  Note: COLMAP should be installed separately")
        print("  Ubuntu: sudo apt install colmap")
        print("  Or from: https://colmap.github.io/install.html")
    print()

    # Summary
    print("=" * 60)
    if all_good and colmap_ok:
        print("✓ All dependencies installed successfully!")
        print()
        print("Next steps:")
        print("  1. Test with a sample video")
        print("  2. Run: python scripts/extract_frames.py <video>")
        print("  3. Run: python scripts/run_colmap.py data/extracted_frames/<video>")
        print("  4. Run: ns-process-data images --data data/extracted_frames/<video> --output-dir data/processed/<video>")
        print("  5. Run: python scripts/train_nerf.py data/processed/<video>")
        return 0
    elif all_good:
        print("⚠ Python packages OK, but COLMAP needs to be installed")
        print()
        print("Install COLMAP:")
        print("  Ubuntu: sudo apt install colmap")
        print("  Or from: https://colmap.github.io/install.html")
        return 1
    else:
        print("✗ Some dependencies are missing")
        print()
        print("Try running:")
        print("  ./setup.sh")
        print("Or:")
        print("  source venv/bin/activate && pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
