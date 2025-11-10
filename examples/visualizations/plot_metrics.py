#!/usr/bin/env python3
"""
Visualize comparison metrics between models
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(results_file: str, output_dir: str = None):
    """
    Create visualization plots from comparison results

    Args:
        results_file: Path to comparison JSON file
        output_dir: Directory to save plots
    """
    # Load results
    with open(results_file, "r") as f:
        data = json.load(f)

    models = data.get("models", [])

    if not models:
        print("No models found in results")
        return

    # Extract metrics
    model_names = []
    psnr_values = []
    ssim_values = []
    fps_values = []
    ms_per_frame_values = []
    training_time_values = []

    for model in models:
        if not model.get("success"):
            continue

        model_names.append(model["model"])
        metrics = model.get("metrics", {})
        timing = model.get("timing", {})
        training_time = model.get("training_time_s", None)

        psnr = metrics.get("psnr", metrics.get("psnr_mean", None))
        ssim = metrics.get("ssim", metrics.get("ssim_mean", None))
        fps = timing.get("fps", None)
        ms_per_frame = timing.get("ms_per_frame", None)

        psnr_values.append(psnr if psnr else 0)
        ssim_values.append(ssim if ssim else 0)
        fps_values.append(fps if fps else 0)
        ms_per_frame_values.append(ms_per_frame if ms_per_frame else 0)
        training_time_values.append(training_time / 60 if training_time else 0)  # Convert to minutes

    if not model_names:
        print("No successful evaluations found")
        return

    # Determine output directory
    if output_dir is None:
        output_dir = Path(results_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Novel View Synthesis Model Comparison", fontsize=16, fontweight='bold')

    # PSNR plot
    if any(psnr_values):
        ax = axes[0, 0]
        bars = ax.bar(model_names, psnr_values, color='steelblue', alpha=0.8)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title("Image Quality (PSNR)", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

    # SSIM plot
    if any(ssim_values):
        ax = axes[0, 1]
        bars = ax.bar(model_names, ssim_values, color='forestgreen', alpha=0.8)
        ax.set_ylabel("SSIM", fontsize=12)
        ax.set_title("Structural Similarity (SSIM)", fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)

    # FPS plot
    if any(fps_values):
        ax = axes[1, 0]
        bars = ax.bar(model_names, fps_values, color='coral', alpha=0.8)
        ax.set_ylabel("FPS", fontsize=12)
        ax.set_title("Rendering Speed (FPS)", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

    # Time per frame plot
    if any(ms_per_frame_values):
        ax = axes[1, 1]
        bars = ax.bar(model_names, ms_per_frame_values, color='mediumpurple', alpha=0.8)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title("Time per Frame", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

    # Training time plot
    if any(training_time_values):
        ax = axes[0, 2]
        bars = ax.bar(model_names, training_time_values, color='darkorange', alpha=0.8)
        ax.set_ylabel("Time (minutes)", fontsize=12)
        ax.set_title("Training Time", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10)

    # Hide unused subplot
    axes[1, 2].axis('off')

    # Rotate x-axis labels if needed
    for ax in axes.flat:
        if ax.get_visible():
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Also create a quality vs speed scatter plot
    if any(psnr_values) and any(fps_values):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(fps_values, psnr_values, s=200, alpha=0.6, c=range(len(model_names)),
                  cmap='viridis', edgecolors='black', linewidth=1.5)

        # Add labels
        for i, name in enumerate(model_names):
            ax.annotate(name, (fps_values[i], psnr_values[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')

        ax.set_xlabel("Rendering Speed (FPS)", fontsize=12)
        ax.set_ylabel("Image Quality (PSNR in dB)", fontsize=12)
        ax.set_title("Quality vs Speed Trade-off", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = output_dir / "quality_vs_speed.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {scatter_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model comparison metrics"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to comparison results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    try:
        plot_comparison(args.results_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
