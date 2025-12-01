import cv2
import os
import shutil
import numpy as np
from pathlib import Path

class ImageProcessor:
    def __init__(self):
        pass

    def calculate_blur_score(self, image):
        """Calculate the Laplacian variance of the image (higher is sharper)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process_images(self, image_dir, max_frames=200, min_frames=50):
        """
        Analyzes images in image_dir, keeps the best ones, and moves rejected ones
        to a 'rejected' subdirectory.
        """
        image_dir = Path(image_dir)
        images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        
        if not images:
            return 0, 0

        print(f"Analyzing {len(images)} images...")
        
        scored_images = []
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                score = self.calculate_blur_score(img)
                scored_images.append((score, img_path))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

        # Sort by score (descending)
        scored_images.sort(key=lambda x: x[0], reverse=True)

        # Determine cutoff
        total_valid = len(scored_images)
        
        if total_valid <= max_frames:
            # If we have fewer than max, keep all valid ones (or filter extremely blurry ones if we had a threshold)
            # For now, we just keep them all if under the limit, unless we want to enforce quality.
            # Let's enforce a loose quality check: drop the bottom 10% if they are significantly worse? 
            # Simpler: just keep all if under max, to ensure coverage.
            keep_count = total_valid
        else:
            # Keep top max_frames
            keep_count = max_frames

        # Ensure we don't drop below min_frames if possible
        keep_count = max(keep_count, min(min_frames, total_valid))
        
        to_keep = set(img_path for _, img_path in scored_images[:keep_count])
        
        # Move rejected
        rejected_dir = image_dir.parent / "rejected_images"
        rejected_dir.mkdir(exist_ok=True)
        
        rejected_count = 0
        for _, img_path in scored_images:
            if img_path not in to_keep:
                try:
                    shutil.move(str(img_path), str(rejected_dir / img_path.name))
                    rejected_count += 1
                except Exception as e:
                    print(f"Error moving {img_path}: {e}")

        print(f"Processing complete. Kept {keep_count}, Rejected {rejected_count}")
        return keep_count, rejected_count
