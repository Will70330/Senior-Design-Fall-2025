#!/usr/bin/env python3
"""
Quick start example for testing the entire pipeline with downloaded datasets
This demonstrates dataloader usage and metrics evaluation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction_pipeline.dataloaders.nerfstudio_dataloader import NerfstudioDataset
from reconstruction_pipeline.evaluation.metrics import MetricsTracker
import numpy as np


def test_dataloader():
    """Test loading the downloaded poster dataset"""
    print("=" * 60)
    print("Testing NeRFStudio Dataloader")
    print("=" * 60)

    # Load poster dataset
    dataset_path = "examples/datasets/nerfstudio/poster"

    print(f"\nLoading dataset from: {dataset_path}")
    dataset = NerfstudioDataset(
        data_dir=dataset_path,
        split="train",
        scale_factor=1.0,
        load_images=True
    )

    print(f"Dataset loaded successfully!")
    print(f"  Number of frames: {len(dataset)}")
    print(f"  Image dimensions: {dataset.image_width}x{dataset.image_height}")
    print(f"  Camera intrinsics:")
    print(f"    fx={dataset.fl_x:.2f}")
    print(f"    fy={dataset.fl_y:.2f}")
    print(f"    cx={dataset.cx:.2f}")
    print(f"    cy={dataset.cy:.2f}")

    # Load a sample frame
    print(f"\nLoading sample frame...")
    sample = dataset[0]

    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Transform matrix shape: {sample['transform_matrix'].shape}")
    print(f"  Intrinsics matrix shape: {sample['intrinsics'].shape}")
    print(f"  File path: {sample['file_path']}")

    return dataset


def test_metrics():
    """Test metrics computation"""
    print("\n" + "=" * 60)
    print("Testing Metrics Tracker")
    print("=" * 60)

    # Create dummy images for testing
    print("\nCreating test images...")
    height, width = 480, 640

    # Target image (ground truth)
    target = np.random.rand(height, width, 3).astype(np.float32)

    # Prediction (with some noise)
    noise = np.random.normal(0, 0.05, (height, width, 3))
    prediction = np.clip(target + noise, 0, 1).astype(np.float32)

    # Compute metrics
    tracker = MetricsTracker()

    print("\nComputing metrics...")
    psnr = tracker.compute_psnr(prediction, target, data_range=1.0)
    print(f"  PSNR: {psnr:.2f} dB")

    try:
        ssim = tracker.compute_ssim(prediction, target, data_range=1.0)
        print(f"  SSIM: {ssim:.4f}")
    except ImportError:
        print("  SSIM: (scikit-image not available)")

    mse = tracker.compute_mse(prediction, target)
    mae = tracker.compute_mae(prediction, target)
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")

    # Test batch evaluation
    print("\nTesting batch evaluation...")
    predictions = [prediction, prediction]
    targets = [target, target]

    metrics = tracker.evaluate_images(predictions, targets, compute_ssim=True)
    print(f"  Batch PSNR mean: {metrics['psnr_mean']:.2f} dB")
    print(f"  Batch PSNR std: {metrics['psnr_std']:.4f}")


def main():
    """Run quick start demo"""
    print("\n" + "=" * 60)
    print("NeRF/3DGS Pipeline Quick Start Demo")
    print("=" * 60)

    # Test dataloader
    try:
        dataset = test_dataloader()
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nMake sure you've downloaded a dataset first:")
        print("  ns-download-data nerfstudio --capture-name poster --save-dir examples/datasets")
        return 1

    # Test metrics
    try:
        test_metrics()
    except Exception as e:
        print(f"\nError computing metrics: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Quick start demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train a model:")
    print("     python scripts/train_nerf.py examples/datasets/nerfstudio/poster")
    print("  2. Compare models:")
    print("     python examples/evaluation/compare_models.py <config1.yml> <config2.yml>")
    print("  3. Visualize results:")
    print("     python examples/visualizations/plot_metrics.py results.json")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
