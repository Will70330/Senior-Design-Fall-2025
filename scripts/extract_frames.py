#!/usr/bin/env python3
"""
CLI script to extract frames from video
Usage: python scripts/extract_frames.py <video_path> [options]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction_pipeline.frame_extraction.video_processor import extract_frames_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Extract optimal frames from video for Structure from Motion"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file (.mp4, .MOV, etc.)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: data/extracted_frames/<video_name>)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum blur score - higher = sharper frames required (default: 100.0)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Maximum similarity with previous frame (default: 0.95)"
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=5,
        help="Minimum frames between selections (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        video_name = Path(args.video_path).stem
        output_dir = Path("data/extracted_frames") / video_name
    else:
        output_dir = Path(args.output)

    print(f"Extracting frames from: {args.video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Settings:")
    print(f"  - Blur threshold: {args.blur_threshold}")
    print(f"  - Similarity threshold: {args.similarity_threshold}")
    print(f"  - Min frame interval: {args.min_interval}")
    print(f"  - Max frames: {args.max_frames or 'unlimited'}")
    print()

    try:
        frame_paths = extract_frames_from_video(
            video_path=args.video_path,
            output_dir=str(output_dir),
            blur_threshold=args.blur_threshold,
            similarity_threshold=args.similarity_threshold,
            min_frame_interval=args.min_interval,
            max_frames=args.max_frames
        )

        print(f"\nSuccess! Extracted {len(frame_paths)} frames")
        print(f"Frames saved to: {output_dir}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
