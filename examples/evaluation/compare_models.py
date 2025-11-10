#!/usr/bin/env python3
"""
Compare multiple trained models (NeRF vs 3DGS) on test datasets
Generates comparison report with PSNR and FPS metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reconstruction_pipeline.evaluation.metrics import MetricsTracker


class ModelComparison:
    """Compare different novel view synthesis models"""

    def __init__(self, output_dir: str = "examples/evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def get_training_time(self, config_path: str) -> Optional[float]:
        """
        Extract training time from model directory

        Args:
            config_path: Path to model config.yml

        Returns:
            Training time in seconds, or None if not available
        """
        config_dir = Path(config_path).parent

        # Check for training stats file
        stats_file = config_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                stats = json.load(f)
                return stats.get("total_training_time_s", None)

        # Check nerfstudio dataparser_transforms.json for timestamp info
        # or estimate from checkpoint timestamps
        checkpoints = list(config_dir.glob("*.ckpt"))
        if checkpoints:
            # Get first and last checkpoint times
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            training_time = checkpoints[-1].stat().st_mtime - checkpoints[0].stat().st_mtime
            return training_time

        return None

    def evaluate_nerfstudio_model(
        self,
        config_path: str,
        output_name: str,
        render_output_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a NeRFStudio model (instant-ngp or splatfacto)

        Args:
            config_path: Path to model config.yml
            output_name: Name for this evaluation
            render_output_path: Where to save rendered images

        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {output_name}...")

        if render_output_path is None:
            render_output_path = self.output_dir / output_name

        # Get training time
        training_time = self.get_training_time(config_path)

        # Run NeRFStudio evaluation (using patched script for PyTorch 2.6 compatibility)
        # Use our patched eval script instead of ns-eval
        eval_script = Path(__file__).parent.parent / "eval_model.py"
        cmd = [
            "python", str(eval_script),
            "--load-config", config_path,
            "--output-path", str(render_output_path / "eval_results.json")
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse evaluation results
            eval_file = render_output_path / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    metrics = json.load(f)

                return {
                    "model": output_name,
                    "config": config_path,
                    "metrics": metrics,
                    "training_time_s": training_time,
                    "success": True
                }
            else:
                print(f"Warning: Evaluation file not found for {output_name}")
                return {"model": output_name, "success": False, "training_time_s": training_time}

        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {output_name}: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return {"model": output_name, "success": False, "error": str(e)}

    def measure_rendering_speed(
        self,
        config_path: str,
        num_frames: int = 100
    ) -> Dict:
        """
        Measure rendering FPS for a model

        Args:
            config_path: Path to model config
            num_frames: Number of frames to render for timing

        Returns:
            Dictionary with timing metrics
        """
        import time

        print(f"Measuring rendering speed...")

        # For NeRFStudio models, we can use ns-render with timing
        temp_output = self.output_dir / "temp_render"
        temp_output.mkdir(exist_ok=True)

        start_time = time.time()

        cmd = [
            "ns-render",
            "dataset",
            "--load-config", config_path,
            "--output-path", str(temp_output / "output.mp4"),
            "--rendered-output-names", "rgb",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()

            total_time = end_time - start_time

            return {
                "total_time_s": total_time,
                "fps": num_frames / total_time if total_time > 0 else 0,
                "ms_per_frame": (total_time / num_frames) * 1000 if num_frames > 0 else 0
            }

        except subprocess.TimeoutExpired:
            print("Rendering timeout")
            return {"error": "timeout"}
        except Exception as e:
            print(f"Error measuring speed: {e}")
            return {"error": str(e)}

    def generate_comparison_report(
        self,
        models: List[Dict],
        report_name: str = "comparison_report"
    ):
        """
        Generate comparison report across models

        Args:
            models: List of model evaluation results
            report_name: Name for the report
        """
        report_path = self.output_dir / f"{report_name}.json"

        # Compile results
        comparison = {
            "models": models,
            "summary": self._generate_summary(models)
        }

        # Save JSON report
        with open(report_path, "w") as f:
            json.dump(comparison, f, indent=2)

        # Generate markdown report
        self._generate_markdown_report(comparison, report_name)

        print(f"\nComparison report saved to: {report_path}")
        print(f"Markdown report: {self.output_dir / f'{report_name}.md'}")

    def _generate_summary(self, models: List[Dict]) -> Dict:
        """Generate summary statistics"""
        summary = {}

        # Extract PSNR values
        psnr_values = {}
        fps_values = {}

        for model in models:
            if model.get("success"):
                metrics = model.get("metrics", {})
                psnr = metrics.get("psnr", metrics.get("psnr_mean", None))
                if psnr:
                    psnr_values[model["model"]] = psnr

                timing = model.get("timing", {})
                fps = timing.get("fps", None)
                if fps:
                    fps_values[model["model"]] = fps

        if psnr_values:
            best_psnr = max(psnr_values.items(), key=lambda x: x[1])
            summary["best_quality"] = {
                "model": best_psnr[0],
                "psnr": best_psnr[1]
            }

        if fps_values:
            best_fps = max(fps_values.items(), key=lambda x: x[1])
            summary["best_speed"] = {
                "model": best_fps[0],
                "fps": best_fps[1]
            }

        return summary

    def _generate_markdown_report(self, comparison: Dict, report_name: str):
        """Generate human-readable markdown report"""
        md_path = self.output_dir / f"{report_name}.md"

        with open(md_path, "w") as f:
            f.write(f"# Novel View Synthesis Model Comparison\n\n")

            # Summary
            summary = comparison.get("summary", {})
            if summary:
                f.write("## Summary\n\n")
                if "best_quality" in summary:
                    f.write(f"**Best Quality:** {summary['best_quality']['model']} "
                           f"(PSNR: {summary['best_quality']['psnr']:.2f} dB)\n\n")
                if "best_speed" in summary:
                    f.write(f"**Best Speed:** {summary['best_speed']['model']} "
                           f"(FPS: {summary['best_speed']['fps']:.2f})\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            f.write("| Model | PSNR (dB) | SSIM | FPS | Time/Frame (ms) | Training Time |\n")
            f.write("|-------|-----------|------|-----|-----------------|---------------|\n")

            for model in comparison["models"]:
                if not model.get("success"):
                    continue

                name = model["model"]
                metrics = model.get("metrics", {})
                timing = model.get("timing", {})
                training_time_s = model.get("training_time_s", None)

                psnr = metrics.get("psnr", metrics.get("psnr_mean", "N/A"))
                ssim = metrics.get("ssim", metrics.get("ssim_mean", "N/A"))
                fps = timing.get("fps", "N/A")
                ms_per_frame = timing.get("ms_per_frame", "N/A")

                if isinstance(psnr, float):
                    psnr = f"{psnr:.2f}"
                if isinstance(ssim, float):
                    ssim = f"{ssim:.4f}"
                if isinstance(fps, float):
                    fps = f"{fps:.2f}"
                if isinstance(ms_per_frame, float):
                    ms_per_frame = f"{ms_per_frame:.2f}"

                # Format training time
                if training_time_s is not None:
                    if training_time_s < 60:
                        training_time = f"{training_time_s:.1f}s"
                    elif training_time_s < 3600:
                        training_time = f"{training_time_s/60:.1f}m"
                    else:
                        training_time = f"{training_time_s/3600:.2f}h"
                else:
                    training_time = "N/A"

                f.write(f"| {name} | {psnr} | {ssim} | {fps} | {ms_per_frame} | {training_time} |\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare NeRF and 3DGS models"
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to model config.yml files"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for each model (optional)"
    )
    parser.add_argument(
        "--measure-speed",
        action="store_true",
        help="Measure rendering speed (slower)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/evaluation/results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Determine names
    if args.names:
        if len(args.names) != len(args.configs):
            print("Error: Number of names must match number of configs")
            return 1
        names = args.names
    else:
        names = [f"model_{i}" for i in range(len(args.configs))]

    # Create comparison
    comparison = ModelComparison(output_dir=args.output_dir)

    # Evaluate each model
    results = []
    for config, name in zip(args.configs, names):
        result = comparison.evaluate_nerfstudio_model(config, name)

        if args.measure_speed and result.get("success"):
            timing = comparison.measure_rendering_speed(config)
            result["timing"] = timing

        results.append(result)

    # Generate report
    comparison.generate_comparison_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
