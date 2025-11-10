#!/usr/bin/env python3
"""
CLI script to train Instant-NGP NeRF with NeRFStudio
Usage: python scripts/train_nerf.py <data_path> [options]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction_pipeline.nerf.instant_ngp_trainer import train_instant_ngp


def main():
    parser = argparse.ArgumentParser(
        description="Train Instant-NGP NeRF using NeRFStudio"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to dataset (NeRFStudio format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: outputs/)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30000,
        help="Maximum training iterations (default: 30000)"
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable web viewer"
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=7007,
        help="Web viewer port (default: 7007)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Instant-NGP NeRF Training")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output or 'outputs/'}")
    print(f"Experiment name: {args.experiment_name or 'auto-generated'}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Web viewer: {'disabled' if args.no_viewer else f'enabled on port {args.viewer_port}'}")
    print("=" * 60)
    print()

    if not args.no_viewer:
        print(f"Web viewer will be available at: http://localhost:{args.viewer_port}")
        print()

    try:
        return_code = train_instant_ngp(
            data_path=args.data_path,
            output_dir=args.output,
            experiment_name=args.experiment_name,
            max_iterations=args.max_iterations,
            enable_viewer=not args.no_viewer,
            viewer_port=args.viewer_port
        )

        if return_code == 0:
            print("\n" + "=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
        else:
            print("\nTraining failed or was interrupted", file=sys.stderr)

        return return_code

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
