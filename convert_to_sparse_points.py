#!/usr/bin/env python3
"""
Utility script to convert existing map_final.ply to sparse_points.ply
for nerfstudio compatibility
"""

import os
import shutil
import glob

def convert_capture(capture_dir):
    """Convert map_final.ply to sparse_points.ply"""
    map_final = os.path.join(capture_dir, "map_final.ply")
    sparse_points = os.path.join(capture_dir, "sparse_points.ply")

    if os.path.exists(map_final) and not os.path.exists(sparse_points):
        shutil.copy2(map_final, sparse_points)
        print(f"✓ Created sparse_points.ply in {os.path.basename(capture_dir)}")
        return True
    elif os.path.exists(sparse_points):
        print(f"⊙ sparse_points.ply already exists in {os.path.basename(capture_dir)}")
        return False
    else:
        print(f"✗ No map_final.ply found in {os.path.basename(capture_dir)}")
        return False

def main():
    # Find all capture directories
    base_dir = os.getcwd()
    capture_dirs = glob.glob(os.path.join(base_dir, "capture_*"))

    if not capture_dirs:
        print("No capture directories found in current directory")
        return

    print(f"Found {len(capture_dirs)} capture directories\n")

    converted = 0
    for capture_dir in sorted(capture_dirs):
        if convert_capture(capture_dir):
            converted += 1

    print(f"\n{'='*60}")
    print(f"Converted {converted} captures")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
