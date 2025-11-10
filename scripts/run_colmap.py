#!/usr/bin/env python3
"""
CLI script to run COLMAP Structure from Motion pipeline
Usage: python scripts/run_colmap.py <image_path> [options]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction_pipeline.sfm.colmap_runner import run_colmap_sfm


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP Structure from Motion pipeline"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: data/sfm_output/<image_dir_name>)"
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default="OPENCV",
        choices=["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "RADIAL", "SIMPLE_RADIAL"],
        help="Camera model (default: OPENCV)"
    )
    parser.add_argument(
        "--matching-mode",
        type=str,
        default="exhaustive",
        choices=["exhaustive", "sequential", "vocab_tree", "spatial"],
        help="Feature matching mode (default: exhaustive)"
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="Use single camera for all images"
    )
    parser.add_argument(
        "--colmap-path",
        type=str,
        default="colmap",
        help="Path to COLMAP executable (default: colmap)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        image_dir_name = Path(args.image_path).name
        output_dir = Path("data/sfm_output") / image_dir_name
    else:
        output_dir = Path(args.output)

    print("=" * 60)
    print("COLMAP Structure from Motion Pipeline")
    print("=" * 60)
    print(f"Image directory: {args.image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Camera model: {args.camera_model}")
    print(f"Matching mode: {args.matching_mode}")
    print(f"Single camera: {args.single_camera}")
    print("=" * 60)
    print()

    try:
        outputs = run_colmap_sfm(
            image_path=args.image_path,
            output_path=str(output_dir),
            camera_model=args.camera_model,
            matching_mode=args.matching_mode,
            single_camera=args.single_camera,
            colmap_executable=args.colmap_path
        )

        print("\n" + "=" * 60)
        print("Success! COLMAP reconstruction complete")
        print("=" * 60)
        print(f"Database: {outputs['database']}")
        print(f"Sparse reconstruction: {outputs['sparse']}")
        if outputs['sparse_txt']:
            print(f"Text format: {outputs['sparse_txt']}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
