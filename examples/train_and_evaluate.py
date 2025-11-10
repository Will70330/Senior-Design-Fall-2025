#!/usr/bin/env python3
"""
Flexible training and evaluation script with dataset selection
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

# Available datasets
NERFSTUDIO_DATASETS = {
    "poster": "poster",
    "bww_entrance": "bww_entrance",
    "storefront": "storefront",
    "vegetation": "vegetation",
    "library": "library",
    "campanile": "campanile",
    "desolation": "desolation",
    "redwoods2": "redwoods2",
    "Egypt": "Egypt",
    "person": "person",
    "kitchen": "kitchen",
    "plane": "plane",
    "dozer": "dozer",
    "floating-tree": "floating-tree",
    "aspen": "aspen",
    "stump": "stump",
    "sculpture": "sculpture",
}

BLENDER_DATASETS = {
    "lego": "lego",
    "chair": "chair",
    "drums": "drums",
    "ficus": "ficus",
    "hotdog": "hotdog",
    "materials": "materials",
    "mic": "mic",
    "ship": "ship",
}

ALL_DATASETS = {**NERFSTUDIO_DATASETS, **BLENDER_DATASETS}


def download_dataset(dataset_name: str, dataset_type: str, base_dir: str = "examples/datasets"):
    """Download a dataset if not already present"""
    dataset_path = Path(base_dir) / dataset_type / dataset_name

    if dataset_path.exists():
        print(f"✓ Dataset {dataset_name} already exists at {dataset_path}")
        return str(dataset_path)

    print(f"Downloading {dataset_name} from {dataset_type}...")

    cmd = [
        "ns-download-data",
        dataset_type,
        "--save-dir", base_dir
    ]

    if dataset_type == "nerfstudio":
        cmd.extend(["--capture-name", dataset_name])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Downloaded {dataset_name}")
        return str(dataset_path)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {dataset_name}")
        print(f"   Error: Google Drive rate limit or download issue")
        print(f"\n   WORKAROUND:")
        print(f"   1. Use --skip-download flag and download manually")
        print(f"   2. Or use a dataset that's already downloaded (poster)")
        print(f"   3. Wait a few minutes and try again")
        print(f"\n   Manual download:")
        if dataset_type == "blender":
            print(f"      Visit: https://github.com/bmild/nerf")
            print(f"      Download: nerf_synthetic.zip")
            print(f"      Extract to: examples/datasets/blender/{dataset_name}/")
        else:
            print(f"      ns-download-data {dataset_type} --capture-name {dataset_name} --save-dir examples/datasets")
        return None


def train_model(
    dataset_path: str,
    model_type: str,
    max_iterations: int,
    output_dir: str,
    extra_args: list = None
):
    """Train a model on the specified dataset"""
    print(f"\nTraining {model_type} on {dataset_path}...")
    print(f"Iterations: {max_iterations}")

    cmd = [
        "ns-train",
        model_type,
        "--data", dataset_path,
        "--max-num-iterations", str(max_iterations),
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", output_dir,
    ]

    if extra_args:
        cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
        return False


def evaluate_model(config_path: str, output_path: str):
    """Evaluate a trained model"""
    print(f"\nEvaluating model: {config_path}")

    eval_script = Path(__file__).parent / "eval_model.py"

    cmd = [
        "python", str(eval_script),
        "--load-config", config_path,
        "--output-path", output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Evaluation complete!")

        # Display results
        if Path(output_path).exists():
            with open(output_path, "r") as f:
                results = json.load(f)
                print("\n" + "=" * 60)
                print("Results:")
                print("=" * 60)
                res = results.get("results", {})
                print(f"PSNR:  {res.get('psnr', 'N/A'):.2f} dB")
                print(f"SSIM:  {res.get('ssim', 'N/A'):.4f}")
                print(f"FPS:   {res.get('fps', 'N/A'):.2f}")
                print(f"LPIPS: {res.get('lpips', 'N/A'):.4f}")
                print("=" * 60)

        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate novel view synthesis models on different datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Datasets:

NeRFStudio Datasets:
  {', '.join(sorted(NERFSTUDIO_DATASETS.keys()))}

Blender Datasets:
  {', '.join(sorted(BLENDER_DATASETS.keys()))}

Examples:
  # Train 3DGS on lego dataset
  python {Path(__file__).name} --dataset lego --model splatfacto --iterations 1000

  # Train Instant-NGP on poster dataset
  python {Path(__file__).name} --dataset poster --model instant-ngp --iterations 5000

  # Train on multiple datasets
  python {Path(__file__).name} --dataset lego chair poster --model splatfacto --iterations 1000

  # High quality training
  python {Path(__file__).name} --dataset vegetation --model splatfacto --iterations 30000 --hq
"""
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        choices=list(ALL_DATASETS.keys()),
        help="Dataset(s) to train on"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="splatfacto",
        choices=["splatfacto", "instant-ngp", "nerfacto"],
        help="Model type to train (default: splatfacto for 3DGS-MCMC)"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)"
    )

    parser.add_argument(
        "--hq",
        action="store_true",
        help="Use high-quality settings (more Gaussians, lower cull threshold)"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (assume already downloaded)"
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"Training Configuration:")
    print(f"  Datasets: {', '.join(args.dataset)}")
    print(f"  Model: {args.model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  High Quality: {args.hq}")
    print("=" * 70)

    # Extra arguments for high quality
    extra_args = []
    if args.hq and args.model == "splatfacto":
        extra_args = [
            "--pipeline.model.cull-alpha-thresh", "0.001",
            "--pipeline.model.densify-grad-thresh", "0.0001"
        ]

    results_summary = []

    for dataset_name in args.dataset:
        print("\n" + "=" * 70)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 70)

        # Determine dataset type
        if dataset_name in NERFSTUDIO_DATASETS:
            dataset_type = "nerfstudio"
        else:
            dataset_type = "blender"

        # Download dataset if needed
        if not args.skip_download:
            dataset_path = download_dataset(dataset_name, dataset_type)
            if not dataset_path:
                print(f"Skipping {dataset_name} due to download failure")
                continue
        else:
            dataset_path = f"examples/datasets/{dataset_type}/{dataset_name}"

        # Verify dataset exists
        if not Path(dataset_path).exists():
            print(f"✗ Dataset not found at {dataset_path}")
            continue

        # Train model
        output_subdir = f"{args.output_dir}/{args.model}_{dataset_name}_i{args.iterations}"
        success = train_model(
            dataset_path,
            args.model,
            args.iterations,
            output_subdir,
            extra_args
        )

        if not success:
            print(f"Skipping evaluation for {dataset_name} due to training failure")
            continue

        # Find config file
        config_files = list(Path(output_subdir).rglob("config.yml"))
        if not config_files:
            print(f"✗ No config.yml found in {output_subdir}")
            continue

        config_path = str(config_files[0])

        # Evaluate if requested
        if not args.skip_eval:
            eval_output = f"{output_subdir}/eval_results.json"
            evaluate_model(config_path, eval_output)

            results_summary.append({
                "dataset": dataset_name,
                "model": args.model,
                "config": config_path,
                "eval_results": eval_output
            })

    # Print summary
    if results_summary:
        print("\n" + "=" * 70)
        print("Summary of All Runs:")
        print("=" * 70)
        for result in results_summary:
            print(f"\n{result['dataset']} ({result['model']}):")
            print(f"  Config: {result['config']}")
            print(f"  Results: {result['eval_results']}")

        print("\n" + "=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print("\n1. Compare models:")
        configs = " ".join([f"\"{r['config']}\"" for r in results_summary])
        names = " ".join([r['dataset'] for r in results_summary])
        print(f"   python examples/evaluation/compare_models.py {configs} --names {names}")

        print("\n2. Visualize results:")
        print(f"   python examples/visualizations/plot_metrics.py results/comparison_report.json")

        print("\n3. Render video from any model:")
        print(f"   ns-render interpolate --load-config \"{results_summary[0]['config']}\" --output-path video.mp4")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
